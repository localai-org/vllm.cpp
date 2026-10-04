#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-gate-commands.py.

A gate command that cannot fail collapses "done" into the implementer's opinion
of its own work. This classifier's only job is to tell a runnable command from
prose that looks like one.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gates = _load("check_gate_commands", "scripts/check-gate-commands.py")


class GatesSectionTests(unittest.TestCase):
    def test_finds_the_gates_heading_at_any_level(self):
        for heading in ("## Gates", "### Gates", "#### Gates and evidence"):
            text = f"# Spec\n\nintro\n\n{heading}\n\nrun `ctest -R foo`\n\n## Next\n\ntail\n"
            section = gates.gates_section(text)
            self.assertIsNotNone(section, heading)
            self.assertIn("ctest", section)
            self.assertNotIn("tail", section, "must stop at the next heading")

    def test_returns_none_when_there_is_no_gates_section(self):
        self.assertIsNone(gates.gates_section("# Spec\n\n## Scope\n\nnothing here\n"))

    def test_is_not_fooled_by_the_word_gates_in_prose(self):
        # "the gates are green" is not a section heading.
        self.assertIsNone(gates.gates_section("# Spec\n\nAll the gates are green.\n"))


class RunnableCommandTests(unittest.TestCase):
    def test_recognises_a_real_command(self):
        for body in [
            "run `ctest -R test_foo`",
            "`python3 scripts/check-agent-record.py`",
            "```\nbash scripts/agent-preflight.sh\n```",
            "`cmake --build build -j`",
            # The repo's MANDATED shape for a gate that touches the GPU. The
            # wrapper quotes the real command, so nothing else can reach it.
            "`flock /tmp/gpu -c 'ctest -R qwen36_paged_engine'`",
            # A built test binary, invoked with no arguments at all.
            "`./build-cuda-121a/tests/test_dropin_abi`",
        ]:
            self.assertTrue(gates.runnable_commands(body), body)

    def test_rejects_prose_that_merely_mentions_gating(self):
        for body in [
            "Correctness, e2e and performance gates apply.",
            "The SACRED gate must pass on GB10.",
            "`docs/BENCHMARKS.md`",
            # Every one of these is on the shipped record and was credited as a
            # runnable command. A backticked FILENAME is not a command, and a
            # tool name must be a whole word: `sha256_cbor` is not `sh`, and
            # `python@3.14` is not `python`.
            "`tests/vllm/models/test_model_registry.cpp`",
            "`tests/`",
            "`sha256_cbor`",
            "`python@3.14`",
            # The GPU lock IDIOM, named in three specs. A wrapper with nothing
            # to run is not a gate; a plain `flock` vocabulary entry credits
            # all three of those rows with a command they do not have.
            "`flock`",
            "`flock /tmp/gpu`",
        ]:
            self.assertEqual(gates.runnable_commands(body), [], body)
        # ...without rejecting a path that really is invoked.
        self.assertTrue(gates.runnable_commands("`scripts/check-agent-record.py`"))
        self.assertTrue(gates.runnable_commands("`./build/vllm-cli --model x`"))

    def test_rejects_a_command_that_cannot_fail(self):
        # These are the exact shapes the spec forbids: a Verify that always
        # succeeds turns "done" into an opinion.
        for body in ["`true`", "`echo ok`", "`:`", "`echo done && true`"]:
            self.assertEqual(gates.runnable_commands(body), [], body)
        # And they are rejected for the RIGHT reason. Without this, the loop
        # above passes vacuously on any implementation that simply fails to
        # recognise `true` as a command at all, and the cannot-fail rule --
        # the point of this classifier -- is pinned by nothing.
        for candidate in ("true", "echo ok", ":", "echo done && true"):
            self.assertTrue(gates.is_command(candidate), candidate)

    def test_rejects_a_piped_command(self):
        # `cmd | tail` reports tail's exit status, so the gate cannot fail.
        self.assertEqual(gates.runnable_commands("`ctest -R foo | tail -5`"), [])


class ShippedRecordTests(unittest.TestCase):
    def test_the_audit_covers_every_gated_state(self):
        self.assertEqual(
            gates.GATED_STATES,
            frozenset({"READY", "ACTIVE", "GATING", "BLOCKED"}),
        )
        # ...and audit() must actually FILTER on it. Asserting the constant's
        # literal value pins nothing about the denominator: deleting the state
        # filter in audit() leaves every other assertion in this file green
        # while the report goes from 97 rows to 726. Task 4 ratchets on that
        # number, so it is pinned here.
        audited = gates.audit()
        self.assertTrue(audited)
        self.assertLessEqual({item["state"] for item in audited}, gates.GATED_STATES)
        # And the filter is only load-bearing if the matrices really do carry
        # rows it excludes -- otherwise the assertion above is vacuous.
        on_record = set()
        for path in gates.AUDITED_MATRIX_PATHS:
            for row in gates.record.parse_claim_rows(path, []):
                on_record.add(row.state)
        self.assertTrue(on_record - gates.GATED_STATES, "filter excludes nothing")

    def test_the_six_lifecycle_matrices_are_audited(self):
        names = {p.name for p in gates.AUDITED_MATRIX_PATHS}
        self.assertIn("feature-matrix.md", names)
        self.assertEqual(len(names), 6)
        # The LIST length too, not just the set of names. If check-agent-record's
        # MATRIX_PATHS ever gains the one appended here, audit() parses that file
        # twice and double-counts every row in it -- the denominator moving
        # silently again, which a set comparison reads as still 6.
        self.assertEqual(len(gates.AUDITED_MATRIX_PATHS), 6)

    def test_sglang_is_excluded_and_the_exclusion_is_justified(self):
        # It was listed and contributed 0 rows: "audited in name only", an
        # absence that reads as a pass (audit artifact risk 6). Dropping it is
        # only honest if it genuinely has no gated rows, so pin THAT rather than
        # the bare absence. sglang-matrix.md carries a CLASSIFICATION column
        # (FUSED / INVENTORIED / NOT-APPLICABLE), not a lifecycle state, so the
        # row parser recognises nothing in it -- and reports no error either.
        # If it ever gains real lifecycle rows this goes red, and the matrix must
        # come back into the audited set rather than stay silently skipped.
        names = {p.name for p in gates.AUDITED_MATRIX_PATHS}
        self.assertNotIn("sglang-matrix.md", names)
        sglang = gates.record.AGENTS / "sglang-matrix.md"
        self.assertTrue(sglang.is_file())
        errors: list[str] = []
        self.assertEqual(gates.record.parse_claim_rows(sglang, errors), [])
        self.assertEqual(errors, [])

    def test_every_record_carries_a_known_verdict(self):
        known = {"runnable", "gates-no-command", "no-gates-section", "no-spec"}
        records = gates.audit()
        self.assertTrue(records)
        for item in records:
            self.assertIn(item["verdict"], known)

    def test_a_matrix_that_does_not_parse_is_a_hard_failure(self):
        # audit() used to build a fresh `errors` list per matrix, hand it to
        # parse_claim_rows and never read it. Stripping rows from a matrix then
        # surfaced as "these baseline rows left the gated population ... re-pin
        # RUNNABLE_BASELINE" -- a parse FAILURE wearing the face of a legitimate
        # record edit, recommending the one action the audit forbids doing
        # blindly. Every mode must go red, --json included.
        original = gates.record.parse_claim_rows

        def broken(path, errors):
            rows = original(path, errors)
            errors.append(f"{path.name}:1: SOME-ROW has 4 cells; header has 6")
            return rows

        gates.record.parse_claim_rows = broken
        try:
            with self.assertRaises(gates.RecordParseError):
                gates.audit()
            noise = io.StringIO()
            with contextlib.redirect_stderr(noise), contextlib.redirect_stdout(noise):
                statuses = [
                    gates.main(argv)
                    for argv in ([], ["--json"], ["--check"], ["--json", "--check"])
                ]
            self.assertEqual(statuses, [1, 1, 1, 1])
        finally:
            gates.record.parse_claim_rows = original

    def test_a_parse_failure_never_reads_as_a_legitimate_record_edit(self):
        # The message is the finding: a broken matrix must not be reported in
        # the words that describe a row correctly leaving the population, and
        # must not steer the reader toward re-pinning the baseline.
        original = gates.record.parse_claim_rows

        def broken(path, errors):
            errors.append(f"{path.name}:1: SOME-ROW must have exactly one canonical state")
            return []

        gates.record.parse_claim_rows = broken
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stderr(buffer):
                self.assertEqual(gates.main(["--check"]), 1)
        finally:
            gates.record.parse_claim_rows = original
        message = buffer.getvalue()
        self.assertIn("did not PARSE", message)
        self.assertIn("must have exactly one canonical state", message)
        self.assertNotIn("left the gated population", message)
        self.assertNotIn("re-pin RUNNABLE_BASELINE in the", message)

    def test_report_mode_exits_zero_even_with_debt(self):
        # 67 of 97 rows cannot state a command today. Report mode must still
        # exit 0 -- the ratchet is step 4, after the debt is recorded.
        self.assertEqual(gates.main([]), 0)


def _bash_array(text: str, name: str) -> list[str]:
    """The entries of a `NAME=(\n ... \n)` array in a bash script.

    Membership in the ARRAY, never a substring of the whole file: preflight
    mentions `check-gate-commands` twice -- once in `CHECKERS`, once in the
    `case` branch that adds `--check` -- so a substring test stays green with
    the CHECKERS entry deleted. That is this repo's recorded defect class (a
    substring `--grep` crediting a row with another row's commits), and it
    would hide the exact mutation step 6 requires to go red.
    """
    match = re.search(rf"(?m)^{re.escape(name)}=\($(.*?)^\)$", text, re.S)
    assert match is not None, f"{name}=( ... ) not found"
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


class RatchetTests(unittest.TestCase):
    def test_the_container_rows_credit_rests_on_scripts_that_exist(self):
        """Why ENG-RELEASE-CONTAINERS is in the baseline, not merely that it is.

        The row was credited on arrival because its spec binds the release chain
        it inherits -- scripts that exist and genuinely fail on a broken staged
        tree. Its OWN gates did not exist when it was pinned. If the credit ever
        came to rest only on scripts the tree does not have, the pin would be a
        certificate for nothing, and this is the case that catches that.
        """
        spec = ROOT / ".agents/specs/container-images.md"
        self.assertTrue(spec.exists(), "the container spec is the source of the credit")
        commands = gates.runnable_commands(spec.read_text(encoding="utf-8"))
        existing = sorted({c for c in commands if (ROOT / c).exists()})
        self.assertTrue(
            existing,
            "every command the container spec cites is absent from the tree; "
            "the runnable credit rests on nothing",
        )
        self.assertIn("scripts/validate-release-archive.py", existing)

    def test_the_baseline_matches_the_shipped_record(self):
        # EXACT equality, in both directions, and that is the whole contract:
        # this is an exact pin, not a shrink-only floor. Lowering the baseline
        # to make a red gate green goes red here, which is the point -- and so
        # does GROWTH. Add a real gate command to a row's spec and `--check`
        # stays 0 while this assertion, preflight and CI go red until the set
        # below is re-pinned. Growth is welcome; silent growth is not.
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_dsa_compose_left_the_gated_population_cleanly(self):
        # MODEL-DSV4-DSA-COMPOSE (#2286) entered the runnable population when its
        # scoping-only spec landed, credited for gates that could fail. The row
        # has since departed the gated population entirely: its record moved to
        # .agents/completed/issue-index.md after the DeepSeek-V4 DSA composition
        # landed through the MODEL-MM-deepseek-v4 vision row (W3 wired the cr==4
        # arm end to end, d4d7e9573), superseding the scoping-only row.
        #
        # Assert the departure on BOTH sides -- gone from the audit AND gone from
        # the baseline -- because a row present in one and not the other is
        # exactly what the exact pin exists to catch (RUNNABLE_BASELINE went
        # red on main with precisely this disagreement until the re-pin).
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertIsNone(verdicts.get("MODEL-DSV4-DSA-COMPOSE"))
        self.assertNotIn("MODEL-DSV4-DSA-COMPOSE", gates.RUNNABLE_BASELINE)

    def test_now_derived_left_the_gated_population_cleanly(self):
        # ENG-NOW-DERIVED (#374) shipped W1-W5 and reached DONE. Closure removes
        # it from both sides of the exact pin; it is not assigned a weaker
        # verdict and its implementation evidence remains in the record.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertIsNone(verdicts.get("ENG-NOW-DERIVED"))
        self.assertNotIn("ENG-NOW-DERIVED", gates.RUNNABLE_BASELINE)

    def test_re_adding_done_to_the_gated_population_breaks_the_pin(self):
        # MUTATION: restoring the departed lifecycle state must expose the five
        # runnable DONE rows and disagree with the re-pinned baseline.
        original = gates.GATED_STATES
        gates.GATED_STATES = frozenset(original | {"DONE"})
        try:
            runnable = {
                r["id"] for r in gates.audit() if r["verdict"] == "runnable"
            }
        finally:
            gates.GATED_STATES = original
        departed = {
            "ENG-ASYNC-SCHED",
            "SERVE-HTTP-TRANSPORT",
            "ENG-NOW-DERIVED",
            "ENG-TRAILER-MERGE-ARTIFACTS",
            "SPEC-BPE-QUADRATIC-MERGE",
            # BACKEND-TENSTORRENT-GDN joined this population when its row moved
            # ACTIVE -> DONE (#1715 close, 2026-08-30): the mutation below must
            # expose it as runnable-DONE and disagree with the re-pinned
            # baseline, exactly like the five rows above.
            "BACKEND-TENSTORRENT-GDN",
        }
        self.assertEqual(runnable - set(gates.RUNNABLE_BASELINE), departed)
        self.assertNotEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_trailer_merge_artifacts_left_the_gated_population_cleanly(self):
        # Credited runnable on arrival at ACTIVE, then gone the same day on
        # reaching DONE (157080c8). Assert the departure on BOTH sides -- gone
        # from the audit AND gone from the baseline -- because a row present in
        # one and not the other is exactly what the exact pin exists to catch.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertIsNone(verdicts.get("ENG-TRAILER-MERGE-ARTIFACTS"))
        self.assertNotIn("ENG-TRAILER-MERGE-ARTIFACTS", gates.RUNNABLE_BASELINE)

    def test_forge_coauthor_is_credited_for_real_commands(self):
        # ENG-FORGE-COAUTHOR (#418) joins the runnable population on arrival, so
        # it earns the credit the same way: its spec's Gates section must name
        # commands that can actually fail, including the per-commit re-check of
        # the real f64f2b71 the row exists to unblock.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-FORGE-COAUTHOR"), "runnable")
        spec = (ROOT / ".agents/specs/forge-coauthor-attribution.md").read_text(
            encoding="utf-8"
        )
        for command in ("scripts/agent-preflight.sh", "agent-integration.py"):
            with self.subTest(command=command):
                self.assertIn(command, spec)

    def test_record_conflict_surfaces_is_credited_for_real_commands(self):
        # ENG-RECORD-CONFLICT-SURFACES (#364) joined the runnable population on
        # arrival, so the credit has to be earned the same way ENG-DOCS-SITE
        # earns it: the spec's Gates section must name commands that can
        # actually fail, not prose. This row's gate is the record gate -- no
        # CUDA, GPU or SACRED gate is implicated because no product source is
        # touched, and the spec says so rather than leaving the absence
        # unexplained.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-RECORD-CONFLICT-SURFACES"), "runnable")
        spec = (ROOT / ".agents/specs/retire-shared-record-surfaces.md").read_text(
            encoding="utf-8"
        )
        for command in ("scripts/agent-preflight.sh", "agent-integration.py"):
            with self.subTest(command=command):
                self.assertIn(command, spec)

    def test_serve_recipe_args_is_credited_for_real_commands(self):
        # SERVE-RECIPE-ARGS (#606) enters the runnable population by leaving
        # SPIKE for ACTIVE, which is the first state that puts it in
        # GATED_STATES at all. The credit is earned the same way the two rows
        # above earn it: the spec's Gates section names commands that can
        # actually fail. The second assertion is the one specific to this row --
        # its gate is CPU-only by construction, so the spec must say why no GPU
        # or oracle leg is implicated rather than leaving the absence to be read
        # as an omission.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("SERVE-RECIPE-ARGS"), "runnable")
        spec = (ROOT / ".agents/specs/serve-recipe-args.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/agent-preflight.sh", spec)
        self.assertIn("No GPU, no oracle run", spec)

    def test_dropping_serve_recipe_args_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved: the entry added
        # for #606 must be what keeps the exact pin agreeing with the audit.
        # Remove it and the equality assertion above has to go red, which is
        # what proves the row was pinned because it entered the population and
        # not to quiet a gate.
        reduced = set(gates.RUNNABLE_BASELINE) - {"SERVE-RECIPE-ARGS"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"SERVE-RECIPE-ARGS"})

    def test_dropping_the_pool_best_fit_row_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved: the entry added
        # for #1922 must be what keeps the exact pin agreeing with the audit.
        # Remove it and the equality assertion above has to go red, which is what
        # proves the row was pinned because it ENTERED the population and not to
        # quiet a gate. Same shape and reason as the two cases beside it.
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-POOL-BEST-FIT"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"ENG-POOL-BEST-FIT"})

    def test_the_pool_best_fit_row_is_credited_for_a_two_arm_gate(self):
        # The entry in RUNNABLE_BASELINE is the WHOLE of what this row changed in
        # the checker, so without a case that reads the SPEC the constant is the
        # only artifact and the credit is plausible rather than checkable --
        # which `scripts/check-pr-size.py`'s `governance_checker` contract
        # refuses.
        #
        # The load-bearing assertion is the LAST one. A gate that only ran the
        # test would be satisfied by a tree in which the borrow does nothing,
        # because the invariant would hold trivially. This row's `## Gates` names
        # BOTH arms of a same-binary A/B -- the default run and the
        # `VT_POOL_BORROW=0` run, whose verdicts must DIFFER -- so a tree with an
        # inert guard cannot satisfy it.
        self.assertIn("ENG-POOL-BEST-FIT", gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-POOL-BEST-FIT"), "runnable")
        spec = (
            gates.ROOT / ".agents/specs/pool-best-fit-retention.md"
        ).read_text(encoding="utf-8")
        # Split on the HEADING, not on the string: other sections name `## Gates`
        # in prose, and splitting on the bare text takes that mention instead of
        # the section.
        gate_section = spec.split("\n## Gates\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn("test_engine_scratch_steady_state", gate_section)
        self.assertIn("ctest --test-dir build", gate_section)
        self.assertIn("VT_POOL_BORROW=0", gate_section)

    def test_dropping_the_ltx2_pin_row_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved: the entry added
        # for #1433 must be what keeps the exact pin agreeing with the audit.
        # Remove it and the equality assertion above has to go red, which is what
        # proves the row was pinned because it ENTERED the population and not to
        # quiet a gate. Same shape and reason as SERVE-REQUEST-LENGTH-GUARD below.
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-UPSTREAM-LTX2-PIN"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"ENG-UPSTREAM-LTX2-PIN"})

    def test_dropping_the_ltx2_vae_row_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved (#1451):
        # `KERNEL-LTX2-VAE` entered the runnable population when its spec grew a
        # `## Gates` section, and the entry added for it must be what keeps the
        # exact pin agreeing with the audit. Remove it and the equality
        # assertion has to go red, which is what proves the row was pinned
        # because it ENTERED the population and not to quiet a gate -- the
        # twelve red RatchetTests that this row's own section rename produced.
        reduced = set(gates.RUNNABLE_BASELINE) - {"KERNEL-LTX2-VAE"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"KERNEL-LTX2-VAE"})

    def test_adding_a_row_that_is_not_runnable_also_breaks_the_pin(self):
        # THE OTHER DIRECTION, and it is the one a removal test cannot cover.
        # A pin that only reds when an entry goes missing would let the set GROW
        # silently, which is how a row that carries no failing command gets
        # credited as gated. `KERNEL-CONV3D` is the honest probe: it is this
        # row's sibling, it is a real KERNEL row, and it is deliberately NOT in
        # the runnable population, so adding it is exactly the mistake the
        # ratchet exists to catch rather than an invented identifier.
        self.assertNotIn("KERNEL-CONV3D", gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertNotEqual(verdicts.get("KERNEL-CONV3D"), "runnable")
        inflated = set(gates.RUNNABLE_BASELINE) | {"KERNEL-CONV3D"}
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, inflated)
        self.assertEqual(inflated - runnable, {"KERNEL-CONV3D"})

    def test_the_ltx2_vae_row_is_credited_for_real_commands(self):
        # The entry in RUNNABLE_BASELINE is the WHOLE of what this row changed
        # in this checker, so without a case that reads the SPEC the constant is
        # the only artifact and the credit is plausible rather than checkable --
        # which `scripts/check-pr-size.py`'s `governance_checker` contract
        # refuses.
        #
        # The load-bearing assertions are the last two: the `## Gates` section
        # must name test binaries that genuinely fail on a broken tree, and must
        # NOT be carried into the population by a `git diff`, which exits 0 in
        # any tree and gates nothing.
        self.assertIn("KERNEL-LTX2-VAE", gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("KERNEL-LTX2-VAE"), "runnable")
        spec = (
            gates.ROOT / ".agents/specs/ltx25-vae-device-residency.md"
        ).read_text(encoding="utf-8")
        # Split on the HEADING, not on the string: other sections name
        # `## Gates` in prose, and splitting on the bare text takes that
        # mention instead of the section.
        gate_section = spec.split("\n## Gates\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn("test_diffusion_device_seam", gate_section)
        self.assertIn("test_ltx2_vae", gate_section)
        self.assertNotIn("git diff", gate_section)

    def test_the_ltx2_pin_row_is_credited_for_real_commands(self):
        # The entry in RUNNABLE_BASELINE is the WHOLE of what this row changed in
        # the checker, so without a case that reads the SPEC the constant is the
        # only artifact and the credit is plausible rather than checkable --
        # which `scripts/check-pr-size.py`'s `governance_checker` contract
        # refuses.
        #
        # A record row can only ever be credited for the checker that reads the
        # record, so the third assertion below is the load-bearing one: the
        # `## Gates` section must name `scripts/check-oracle-pins.py`, which
        # genuinely fails on a broken record, rather than a command that exits 0
        # in any tree.
        self.assertIn("ENG-UPSTREAM-LTX2-PIN", gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-UPSTREAM-LTX2-PIN"), "runnable")
        spec = (
            gates.ROOT / ".agents/specs/oracle-ltx-2-pin.md"
        ).read_text(encoding="utf-8")
        # Split on the HEADING, not on the string: other sections name
        # `## Gates` in prose, and splitting on the bare text takes that
        # mention instead of the section.
        gate_section = spec.split("\n## Gates\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn("scripts/check-oracle-pins.py", gate_section)
        self.assertIn("scripts/check-agent-record.py", gate_section)

    def test_dropping_the_request_length_guard_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved: the entry added
        # for #1541 must be what keeps the exact pin agreeing with the audit.
        # Remove it and the equality assertion above has to go red, which is
        # what proves the row was pinned because it ENTERED the population and
        # not to quiet a gate. Same shape and reason as SERVE-RECIPE-ARGS above.
        reduced = set(gates.RUNNABLE_BASELINE) - {"SERVE-REQUEST-LENGTH-GUARD"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"SERVE-REQUEST-LENGTH-GUARD"})

    def test_the_request_length_guard_is_credited_for_real_commands(self):
        # The entry in RUNNABLE_BASELINE is the WHOLE of what this row changed
        # in the checker, so without a case that reads the SPEC the constant is
        # the only artifact and the credit is plausible rather than checkable --
        # which `scripts/check-pr-size.py`'s `governance_checker` contract
        # refuses, and it refused this row's first commit by name.
        #
        # Three things are pinned, not one: the row is in the exact pin, the
        # audit really verdicts it `runnable`, and its `## Gates` section really
        # does name commands that can fail rather than a `git diff --stat` that
        # exits 0 unconditionally.
        self.assertIn("SERVE-REQUEST-LENGTH-GUARD", gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("SERVE-REQUEST-LENGTH-GUARD"), "runnable")
        spec = (
            gates.ROOT / ".agents/specs/serve-request-length-guard.md"
        ).read_text(encoding="utf-8")
        # Split on the HEADING, not on the string: the spec's `## Now`
        # section names `## Gates` in prose, and splitting on the bare
        # text takes that mention instead of the section.
        gate_section = spec.split("\n## Gates\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn("ninja -C build", gate_section)
        self.assertIn("./build/tests/test_openai_api_server", gate_section)

    def test_residency_config_is_credited_for_real_commands(self):
        # ENG-RESIDENCY-CONFIG (#1110) is a NEW row arriving at ACTIVE, which is
        # the first state that puts it in GATED_STATES at all, so it joins the
        # runnable population on arrival and earns the credit the same way the
        # rows above earn it. The entry in RUNNABLE_BASELINE is the WHOLE of what
        # this row changed in the checker, so without these two cases the
        # constant is the only artifact and the credit is plausible rather than
        # checkable -- which is what `scripts/check-pr-size.py`'s
        # `governance_checker` contract refuses, and it refused this row's first
        # commit by name.
        #
        # THREE things are pinned, not one. The row is in the exact pin; its
        # Gates section really does yield a command that can fail; and the spec
        # says why no GPU, oracle or throughput leg is implicated, because a
        # CPU-only credit whose record is silent about the missing arms reads as
        # coverage. This row's claim for that silence is unusually strong and is
        # worth holding: it moves where a value comes from and changes no kernel,
        # dtype, allocation or token, so there is no throughput axis to gate.
        self.assertIn("ENG-RESIDENCY-CONFIG", gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-RESIDENCY-CONFIG"), "runnable")

        spec = (ROOT / ".agents/specs/weight-residency-config.md").read_text(
            encoding="utf-8"
        )
        section = gates.gates_section(spec)
        self.assertIsNotNone(section)
        # EXACT strings, because `runnable_commands` returns whole commands and
        # `assertIn` on a list is membership rather than a substring search. The
        # first draft of this case asserted the bare `scripts/agent-preflight.sh`
        # and went red for that reason, not because the spec lacked a gate.
        commands = gates.runnable_commands(section)
        self.assertIn("scripts/agent-preflight.sh --staged", commands)
        self.assertIn("ctest --test-dir build -j 6", commands)
        # No throughput axis, said in the record rather than left as an absence.
        self.assertIn("no throughput axis of its own and", spec)
        # And the GPU leg that IS owed is recorded as owed, not skipped.
        self.assertIn("## Owed", spec)
        self.assertIn("370 GiB", spec)

    def test_dropping_residency_config_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved: the entry added
        # for #1110 must be what keeps the exact pin agreeing with the audit.
        # Remove it and set equality has to go red, which is what proves the row
        # was pinned because it entered the population and not to quiet a gate.
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-RESIDENCY-CONFIG"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"ENG-RESIDENCY-CONFIG"})
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_mtp_depth_is_credited_for_real_commands(self):
        # SPEC-MTP-K-GT-1 (#81) is a NEW row arriving at ACTIVE, which is the
        # first state that puts it in GATED_STATES at all, so it joins the
        # runnable population on arrival and earns the credit the same way the
        # three rows above earn it. TWO things are pinned here rather than one:
        # the row is in the exact pin, AND its Gates section really does yield a
        # command that can fail. A row pinned without the second half is a
        # certificate for nothing, which is the failure the checker's own header
        # admits to for five older credits.
        self.assertIn("SPEC-MTP-K-GT-1", gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("SPEC-MTP-K-GT-1"), "runnable")

        spec = (ROOT / ".agents/specs/mtp-k-gt-1.md").read_text(encoding="utf-8")
        section = gates.gates_section(spec)
        self.assertIsNotNone(section)
        self.assertIn("scripts/agent-preflight.sh", gates.runnable_commands(section))
        # The row's GPU leg is OWED, not skipped, and the spec has to say so.
        # Without this the credit could rest on a CPU gate while the record
        # stayed silent about the arm nobody ran, which reads as coverage.
        self.assertIn("DGX three-way greedy gate at k=2, 3, 4", spec)

    def test_dropping_mtp_depth_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved: the entry added
        # for #81 must be what keeps the exact pin agreeing with the audit.
        # Remove it and set equality has to go red, which is what proves the row
        # was pinned because it entered the population and not to quiet a gate.
        reduced = set(gates.RUNNABLE_BASELINE) - {"SPEC-MTP-K-GT-1"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"SPEC-MTP-K-GT-1"})
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_mtp_depth_losing_its_command_is_refused_by_name(self):
        # The OTHER direction the same set equality binds. The case above moves
        # the BASELINE and holds the record still; this one moves the RECORD and
        # holds the baseline still, which is the half `ratchet_errors` owns and
        # the half that ships in CI. Only a row the baseline already names can
        # be reported as having LOST its command, so a pin taken without this
        # row would report nothing at all here -- a regression on a row nobody
        # pinned is silent, not green.
        records = [dict(item) for item in gates.audit()]
        victim = next(item for item in records if item["id"] == "SPEC-MTP-K-GT-1")
        self.assertEqual(victim["verdict"], "runnable")
        victim["verdict"] = "gates-no-command"
        errors = gates.ratchet_errors(records)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("SPEC-MTP-K-GT-1", errors[0])
        self.assertIn("Repair the row", errors[0])

    def test_the_baseline_re_pin_is_load_bearing(self):
        # MUTATION: the re-pin that added this row must be what makes the audit
        # agree with the baseline. Drop the entry and the exact-pin assertion
        # above has to go red -- otherwise the baseline is decorative and a row
        # could enter or leave the runnable population unnoticed, which is the
        # failure the exact pin exists to catch.
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-RECORD-CONFLICT-SURFACES"}
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(
            runnable,
            reduced,
            "removing the row from the baseline must break the pin",
        )
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_dropping_the_anchor_ratchet_from_the_pin_breaks_it(self):
        # MUTATION for the 2026-08-14 re-pin (#632). ENG-RECORD-ANCHOR-RATCHET
        # entered the runnable population when the row left SPIKE for ACTIVE, so
        # the entry added for it must be what keeps the exact pin agreeing with
        # the audit. Remove it and the equality assertion has to go red, which is
        # what proves the row was pinned because it entered the population and
        # not to quiet a gate.
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-RECORD-ANCHOR-RATCHET"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"ENG-RECORD-ANCHOR-RATCHET"})

    def test_the_anchor_ratchet_credit_is_its_own_gate(self):
        # The credit has to be EARNED, not inherited: unlike the weak credits the
        # RUNNABLE_BASELINE header admits to, this row's gate IS the checker its
        # spec names, so the credited command is the thing under test. Pin both
        # halves -- the row audits runnable, and the invocation its Gates section
        # carries is the one that reds on either direction of the ratchet. A spec
        # rewritten into prose gates goes red here rather than keeping a credit
        # it no longer deserves.
        row = "ENG-RECORD-ANCHOR-RATCHET"
        self.assertIn(row, gates.RUNNABLE_BASELINE)
        record = next(r for r in gates.audit() if r["id"] == row)
        self.assertEqual(record["verdict"], "runnable", record)

        section = gates.gates_section(
            (ROOT / ".agents/specs/record-anchor-ratchet.md").read_text(encoding="utf-8")
        )
        self.assertIsNotNone(section)
        commands = gates.runnable_commands(section)
        self.assertIn("python3 scripts/check-agent-record.py --report", commands)
        self.assertIn("python3 tests/scripts/test_agent_record.py", commands)

    def test_eng_docs_site_is_credited_for_real_commands(self):
        # ENG-DOCS-SITE joined the runnable population on arrival rather than
        # being parked as gates-no-command, so the credit has to be earned by
        # the spec actually naming commands that can fail. Pin that: the row is
        # runnable AND its Gates section carries the two that do the work. A
        # spec rewritten into prose gates goes red here rather than silently
        # keeping a credit it no longer deserves.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-DOCS-SITE"), "runnable")

        spec = ROOT / ".agents" / "specs" / "gh-pages-docs-site.md"
        body = spec.read_text(encoding="utf-8")
        self.assertIn("python3 scripts/check-site.py", body)
        self.assertIn("hugo --minify -s website", body)

    def test_the_newly_pinned_row_carries_a_reproducible_invocation(self):
        # The baseline is hand-edited and `audit()` is not, so the failure mode a
        # re-pin invites is adding an id whose spec does not actually tell anyone
        # how to run the gate. Set-equality above cannot catch that: both sides
        # move together.
        #
        # It is worth being precise about what this does and does not prove. The
        # obvious mutation -- strip the fenced block, expect the credit to vanish
        # -- does NOT hold, and finding that out is the useful part: with both
        # blocks removed the section still yields `ctest`, harvested from an
        # inline code span in the surrounding prose. That is the weak-credit debt
        # the checker's own header already admits to (see the RUNNABLE_BASELINE
        # comment and gate-command-audit-2026-08-06.md risk 3), not something
        # this row introduced, so it is recorded here rather than worked around.
        #
        # What is asserted instead is the thing a reader actually needs: the row
        # is pinned, it audits runnable, and its Gates section carries a real
        # fenced invocation naming the binaries the gate was run with.
        row = "SAMPLE-PROMPT-LOGPROBS"
        self.assertIn(row, gates.RUNNABLE_BASELINE)
        record = next(r for r in gates.audit() if r["id"] == row)
        self.assertEqual(record["verdict"], "runnable", record)

        section = gates.gates_section(
            (ROOT / ".agents/specs/prompt-logprobs.md").read_text(encoding="utf-8")
        )
        self.assertIsNotNone(section)
        blocks = re.findall(r"```sh\n(.*?)```", section, flags=re.DOTALL)
        self.assertTrue(blocks, "the Gates section carries no fenced invocation")
        body = "\n".join(blocks)
        for fragment in ("cmake -S . -B", "cmake --build", "test_llm_engine", "ctest"):
            self.assertIn(fragment, body, fragment)

    def test_lora_runtime_is_credited_for_its_own_invocation(self):
        # LORA-RUNTIME re-entered the runnable population when the row went back
        # to ACTIVE for W2 (#278), and the credit it arrived with was WEAK: the
        # UPSTREAM path `tests/lora/test_qwen35_densemodel_lora.py`, named in
        # the spec's prose as the eventual model gate, which nothing here runs.
        # The pin was taken only alongside the row's real CPU invocation, so
        # assert the real one is what the spec carries. Delete the ctest lines
        # and this goes red, instead of the row quietly keeping a credit that
        # rests on an upstream filename.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("LORA-RUNTIME"), "runnable")

        spec = ROOT / ".agents" / "specs" / "lora-adapter.md"
        section = gates.gates_section(spec.read_text(encoding="utf-8"))
        self.assertIsNotNone(section)
        commands = gates.runnable_commands(section)
        self.assertIn(
            "ctest --test-dir build-cpu -R test_lora_layers --output-on-failure",
            commands,
        )
        self.assertIn(
            "ctest --test-dir build-cpu -R test_punica_cpu --output-on-failure",
            commands,
        )

    def test_a_row_that_loses_its_command_is_refused(self):
        # Still present, still gated, no longer runnable -- a real regression.
        victim = sorted(gates.RUNNABLE_BASELINE)[0]
        records = [{"verdict": "gates-no-command", "id": victim, "state": "READY",
                    "path": "p", "line": 1, "detail": "d"}]
        errors = gates.ratchet_errors(records)
        self.assertTrue(errors)
        self.assertIn(victim, errors[0])
        self.assertIn("Repair the row", errors[0])

    def test_a_row_that_left_the_population_reports_differently(self):
        # Deleted or transitioned out: legitimate, but must re-pin. The two
        # cases MUST be distinguishable -- that is why the baseline is a set.
        errors = gates.ratchet_errors([])
        self.assertTrue(errors)
        self.assertTrue(any("left the gated population" in e for e in errors))
        self.assertFalse(any("Repair the row" in e for e in errors))

    def test_a_lost_row_and_a_departed_row_are_reported_separately(self):
        # The two cases in ONE run, which is the only arrangement that proves
        # they are distinguishable rather than merely differently worded. The
        # two tests above each see a single case, so both stay green if the
        # `& present` split is deleted and every drop is called a loss; here
        # the departed row would then be named in the "Repair the row" message
        # and the assertion below goes red.
        ordered = sorted(gates.RUNNABLE_BASELINE)
        self.assertGreaterEqual(len(ordered), 2)
        lost, departed = ordered[0], ordered[1]
        records = [{"verdict": "gates-no-command", "id": lost, "state": "READY",
                    "path": "p", "line": 1, "detail": "d"}]
        records += [
            {"verdict": "runnable", "id": rid, "state": "READY",
             "path": "p", "line": 2, "detail": "d"}
            for rid in ordered[2:]
        ]
        errors = gates.ratchet_errors(records)
        self.assertEqual(len(errors), 2, errors)
        loss_msg = [e for e in errors if "Repair the row" in e]
        gone_msg = [e for e in errors if "left the gated population" in e]
        self.assertEqual(len(loss_msg), 1, errors)
        self.assertEqual(len(gone_msg), 1, errors)
        self.assertIn(lost, loss_msg[0])
        self.assertNotIn(departed, loss_msg[0])
        self.assertIn(departed, gone_msg[0])
        self.assertNotIn(lost, gone_msg[0])

    def test_an_improvement_is_allowed(self):
        records = [
            {"verdict": "runnable", "id": rid, "state": "READY",
             "path": "p", "line": 1, "detail": "d"}
            for rid in sorted(gates.RUNNABLE_BASELINE)
        ] + [{"verdict": "runnable", "id": "NEW-ROW", "state": "READY",
              "path": "p", "line": 2, "detail": "d"}]
        self.assertEqual(gates.ratchet_errors(records), [])

    def test_check_mode_passes_on_the_shipped_record(self):
        # The gate ships GREEN. It was wired after the debt was recorded, so it
        # never had to be relaxed to pass.
        self.assertEqual(gates.main(["--check"]), 0)

    def test_check_mode_is_never_silently_swallowed_by_json(self):
        # --json used to be the only mode, and it returns 0 unconditionally. If
        # it short-circuits ahead of --check, `--json --check` is a gate that
        # CANNOT FAIL -- the precise shape this classifier exists to detect,
        # wearing this classifier's own face.
        self.assertEqual(gates.main(["--json", "--check"]), 0)
        original = gates.RUNNABLE_BASELINE
        try:
            gates.RUNNABLE_BASELINE = frozenset(original | {"ROW-THAT-IS-NOT-THERE"})
            self.assertEqual(gates.main(["--check"]), 1)
            self.assertEqual(gates.main(["--json", "--check"]), 1)
        finally:
            gates.RUNNABLE_BASELINE = original

    def test_the_checker_is_wired_into_preflight_and_ci(self):
        preflight = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        self.assertIn("check-gate-commands", _bash_array(preflight, "CHECKERS"))
        self.assertIn("test_check_gate_commands", _bash_array(preflight, "SUITES"))
        # ...and dispatched WITH --check. Report mode exits 0 whatever the
        # record says, so a CHECKERS entry without the flag installs a gate
        # that cannot fail.
        self.assertRegex(
            preflight, r"(?m)^\s*[\w|-]*check-gate-commands\)\s+run\b.*--check"
        )
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/check-gate-commands.py --check", ci)
        self.assertIn("tests/scripts/test_check_gate_commands.py", ci)

    def test_cudagraph_dedup_is_credited_for_real_commands(self):
        # ENG-CUDAGRAPH-DEDUP (#1162) is a NEW row arriving at ACTIVE, which is
        # the first state that puts it in GATED_STATES at all, so it joins the
        # runnable population on arrival. TWO things are pinned, not one: the
        # row is in the exact pin, AND its Gates section really does yield a
        # command that can fail. A row pinned without the second half is a
        # certificate for nothing, which is the failure the checker's own header
        # admits to for five older credits.
        self.assertIn("ENG-CUDAGRAPH-DEDUP", gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-CUDAGRAPH-DEDUP"), "runnable")

        spec = (ROOT / ".agents/specs/eng-cudagraph-dedup.md").read_text(encoding="utf-8")
        section = gates.gates_section(spec)
        self.assertIsNotNone(section)
        commands = gates.runnable_commands(section)
        # The credit rests on the focused suite, not on the preflight line: that
        # suite detected 9 of 9 negative mutations of the registry this row adds,
        # so it genuinely goes red when the row regresses. `git diff --stat` is
        # also extracted from the mutation bullet and is one of the weak credits
        # this checker's header names; it is deliberately not what is asserted.
        self.assertIn("ctest -R test_graph_dedup", commands)
        self.assertIn("./scripts/agent-preflight.sh", commands)

        # BOTH credited commands are CPU-tier, so the credit says nothing about
        # the DEVICE tier, and the record must not go silent about the leg the
        # credit does not cover. That was the point of the assertion that used to
        # sit here, and it read the SENTENCE: `"Device byte-identity A/B (owed"`.
        # fe24a3029 legitimately rewrote that sentence when the A/B ran, and took
        # main red (#1229). `owed` was the leg's value on the day, not the
        # property. Re-pinning onto `RAN 2026-08-18, PASS` would only move the pin
        # and the rerun owed under #1226 would red it again, so this keys on what a
        # correct record edit PRESERVES: the numbered item, its SUBJECT, and
        # whether the checker's own extractor finds a command in it.
        items = gates.gate_items(section)
        device = [it for it in items if "byte-identity" in (gates.item_lead(it) or "")]
        # Uniqueness, not existence: a positional match would silently move onto a
        # second device gate if the row ever grows one.
        self.assertEqual(len(device), 1, [gates.item_lead(i) for i in items])
        # STRUCTURAL, and the load-bearing half: the classifier itself extracts
        # nothing runnable from this item, which is what proves the credit above
        # rests entirely on the CPU tier rather than on a reader's belief.
        self.assertEqual(gates.runnable_commands(device[0]), [])
        # ...and the record therefore has to say what became of it. Satisfied by
        # `owed` before fe24a3029 and by `RAN` after it, which is the whole point.
        self.assertIsNotNone(gates.gate_disposition(device[0]))

    def test_gate_disposition_reads_the_lead_not_the_body(self):
        # Why the LEAD and not the whole item. A status word in the lead is a
        # declaration ABOUT the gate; the same word in the body is ordinary prose
        # describing what the gate does. Widen the search and the anti-silence
        # rule above starts crediting items that declare nothing.
        #
        # This is MEASURED, and the first measurement REFUTED the reason this
        # test was originally written with. eng-cudagraph-dedup.md was the claimed
        # example and it is not one: all 7 of its gate items agree under both
        # scopes, so widening the scope leaves the assertion above green and that
        # spec cannot pin this. Surveyed instead over every `Gates` section in
        # .agents/specs/ on 2026-08-18: 32 of 323 gate items GAIN a disposition
        # when the search widens, among them `**No regression:**` in
        # cpu-elementwise-gemm.md and `**Correctness gate:**` in dropin-kernel-abi.md,
        # both of which pick up `pass` out of body prose.
        #
        # The fixture is that shape, written out rather than read from another
        # row's spec: pinning a live sentence in a file this row does not own is
        # the defect #1229 exists to remove, and it would put this suite back in
        # the path of somebody else's correct record edit.
        body_only = "3. **Correctness gate:** the ported cases all pass on CPU."
        self.assertIsNone(gates.gate_disposition(body_only))
        self.assertIsNotNone(gates._DISPOSITION.search(body_only))

        # STATE-INDEPENDENCE, which is what makes this a repair and not a re-pin.
        # Both wordings the device leg has ever carried are fixtures here, quoted
        # from 2a976eb9f and fe24a3029. The assertion above holds on BOTH, so it is
        # not describing today's sentence.
        owed = "6. **Device byte-identity A/B (owed, see below).** Same binary."
        ran = (
            "6. **Device byte-identity A/B \u2014 RAN 2026-08-18, PASS; see "
            "[`## Outcome`](#outcome).** Same binary."
        )
        self.assertEqual(gates.gate_disposition(owed), "owed")
        self.assertEqual(gates.gate_disposition(ran), "RAN")
        self.assertEqual(gates.runnable_commands(owed), [])
        self.assertEqual(gates.runnable_commands(ran), [])

        # A lead stripped of its status is the SILENCE this rule exists to catch,
        # and it must read as undeclared rather than as some default.
        silent = "6. **Device byte-identity A/B.** Same binary."
        self.assertIsNone(gates.gate_disposition(silent))
        # An item with no bold lead at all is undeclared too, never green by
        # accident: an unparsed record takes the gate RED, not past it.
        self.assertIsNone(gates.item_lead("6. Device byte-identity A/B, owed."))
        self.assertIsNone(gates.gate_disposition("6. Device byte-identity A/B, owed."))

    def test_dropping_cudagraph_dedup_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved: the entry added
        # for #1162 must be what keeps the exact pin agreeing with the audit.
        # Remove it and set equality has to go red, which is what proves the row
        # was pinned because it entered the population and not to quiet a gate.
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-CUDAGRAPH-DEDUP"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"ENG-CUDAGRAPH-DEDUP"})
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_dropping_cudagraph_break_from_the_pin_breaks_it(self):
        # MUTATION for the #1376 repair. ENG-CUDAGRAPH-BREAK entered the runnable
        # population when W5 (#1361) filled its spec's Gates section with runnable
        # evidence, and the re-pin that change owed was not made, so main itself
        # failed this suite 8 times of 44. Remove the entry and set equality has
        # to go red, which is what proves the row was pinned because it entered
        # the population rather than to quiet a gate.
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-CUDAGRAPH-BREAK"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"ENG-CUDAGRAPH-BREAK"})
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_dropping_exl3_perf_from_the_pin_breaks_it(self):
        # MUTATION for the QUANT-EXL3-PERF re-pin (#2570). The row is new and it
        # entered the runnable population the moment its spec carried a `## Gates`
        # section naming commands that can FAIL -- two `ctest` invocations and
        # `scripts/agent-preflight.sh --staged`. Remove the entry and set equality
        # has to go red, which is what proves the pin was taken BECAUSE the row
        # entered the population rather than to quiet a gate.
        reduced = set(gates.RUNNABLE_BASELINE) - {"QUANT-EXL3-PERF"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"QUANT-EXL3-PERF"})
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_exl3_perf_gates_can_actually_fail(self):
        # The population is defined by what a Gates section can RUN, and a
        # `## Gates` block whose only command is a `git diff` gates nothing --
        # this tree has shipped one. So assert the CONTENT, not just membership:
        # the row's gates name a real ctest target that exists in the tree, and
        # the suite it names is the one this row's arm is measured by.
        spec = ROOT / ".agents/specs/quant-exl3-perf.md"
        text = spec.read_text(encoding="utf-8")
        self.assertIn("test_exl3_gemv", text)
        self.assertIn("ctest", text)
        # The named suite must EXIST, or the gate is a string.
        self.assertTrue((ROOT / "tests/vt/test_exl3_gemv.cpp").is_file())
        # And it must be registered, or `ctest -R` matches nothing and exits 0
        # having run no test at all -- a gate that cannot run must not pass.
        cmake = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("test_exl3_gemv", cmake)

    def test_preflight_compiles_left_the_gated_population_cleanly(self):
        # b39f14adb moves ENG-PREFLIGHT-COMPILES (#2401) ACTIVE -> PARTIAL.
        # PARTIAL is outside GATED_STATES, so the row must leave both sides of
        # the exact pin without losing the runnable commands preserved in its
        # spec. These three assertions distinguish lifecycle shrinkage from a
        # weaker verdict or a silently deleted gate command.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertIsNone(verdicts.get("ENG-PREFLIGHT-COMPILES"))
        self.assertNotIn("ENG-PREFLIGHT-COMPILES", gates.RUNNABLE_BASELINE)
        spec = ROOT / ".agents/specs/preflight-compiles.md"
        text = spec.read_text(encoding="utf-8")
        section = gates.gates_section(text)
        self.assertIsNotNone(section)
        commands = gates.runnable_commands(section)
        self.assertIn("python3 tests/scripts/test_check_tree_compiles.py", commands)
        self.assertIn(
            "python3 scripts/check-tree-compiles.py --base origin/main", commands
        )

    def test_hf_model_download_earns_its_runnable_baseline_entry(self):
        # ENG-HF-MODEL-DOWNLOAD (#1280) arrives at READY, so it enters the gated
        # population for the first time and the baseline grows by one. Same
        # shape and same reason as the row above.
        #
        # The row is classified runnable because its spec's `## Gates` section
        # names `scripts/validate-container-image.py`, a command that can fail.
        # The exact pin proves the SET agrees, which holds for any membership
        # and cannot say this row belongs. This says it: remove the entry and
        # set equality has to go red, which is what separates a row pinned
        # because it entered the population from a row pinned to quiet a gate.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-HF-MODEL-DOWNLOAD"), "runnable")
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-HF-MODEL-DOWNLOAD"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"ENG-HF-MODEL-DOWNLOAD"})
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_backend_tenstorrent_gdn_left_the_gated_population_cleanly(self):
        # BACKEND-TENSTORRENT-GDN (#1715) entered the runnable population on
        # 2026-08-22 as inherited credit while its spec was committed and its
        # W1/W2 waves were still owed. The row's code landed (W1 `34fde3502`,
        # W2 `c85af0aaf`), the Qwen3_5 wiring row made the family
        # production-reached, and the record-only close moved the row
        # ACTIVE -> DONE. `DONE` is not in GATED_STATES, so it leaves the
        # AUDITED population entirely.
        #
        # Assert the departure on BOTH sides -- gone from the audit AND gone
        # from the baseline -- because a row present in one and not the other
        # is exactly what the exact pin exists to catch. Same shape and same
        # reason as SPEC-BPE-QUADRATIC-MERGE below.
        #
        # The row is NOT re-verdicted downward and it did not lose its command:
        # the suites it credited are still pinned on the live rows that bind
        # them. This case pins the difference, since a row that leaves and a
        # row that silently drops its gate command are the two things this
        # classifier must never confuse.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertIsNone(verdicts.get("BACKEND-TENSTORRENT-GDN"))
        self.assertNotIn("BACKEND-TENSTORRENT-GDN", gates.RUNNABLE_BASELINE)

    def test_bpe_quadratic_merge_left_the_gated_population_cleanly(self):
        # SPEC-BPE-QUADRATIC-MERGE (#1365) entered the runnable population on
        # 2026-08-19 as growth, when its `## Gates` section gained the `g++`
        # build and the run lines for `tools/bench/bpe_encode_cost.cpp`. Its
        # closing commit promoted the row `GATING` -> `DONE`, and `DONE` is not
        # in GATED_STATES, so it leaves the AUDITED population entirely.
        #
        # Assert the departure on BOTH sides -- gone from the audit AND gone
        # from the baseline -- because a row present in one and not the other is
        # exactly what the exact pin exists to catch. Same shape and same reason
        # as ENG-TRAILER-MERGE-ARTIFACTS and ENG-NOW-DERIVED above.
        #
        # The row is NOT re-verdicted downward and it did not lose its command:
        # the recipe is still in the spec. This case pins the difference, since
        # a row that leaves and a row that silently drops its gate command are
        # the two things this classifier must never confuse.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertIsNone(verdicts.get("SPEC-BPE-QUADRATIC-MERGE"))
        self.assertNotIn("SPEC-BPE-QUADRATIC-MERGE", gates.RUNNABLE_BASELINE)


if __name__ == "__main__":
    unittest.main()


class Qwen35GdnRunnablePopulationTests(unittest.TestCase):
    """The BACKEND-TENSTORRENT-QWEN35 entry is the whole of what #1715's wiring
    row changed in this checker, so the credit has to be checkable rather than
    plausible: the row must be IN the pinned population, its audit verdict must
    be earned by the record (not merely asserted), and removing the entry must
    break the exact pin -- which is what proves the row was pinned because it
    ENTERED the runnable population and not to quiet a gate."""

    ROW = "BACKEND-TENSTORRENT-QWEN35"

    def test_the_row_is_pinned_and_its_verdict_is_runnable(self):
        self.assertIn(self.ROW, gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get(self.ROW), "runnable")

    def test_dropping_the_qwen35_gdn_entry_breaks_the_pin(self):
        # MUTATION, in the direction this pin actually moved (#1715): remove the
        # entry and the exact-population equality inside the checker has to go
        # red, because the audit still reports the row as runnable from its spec.
        reduced = set(gates.RUNNABLE_BASELINE) - {self.ROW}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {self.ROW})

    def test_the_qwen35_gdn_row_is_credited_for_real_commands(self):
        # A record row can only be credited for the checker that reads the
        # record, so the load-bearing assertion is the last one: the `## Gates`
        # section of the row's own spec must name commands that genuinely fail
        # on a broken tree -- the parity binary and the preflight chain -- not
        # a command that exits 0 in any tree (`git diff`).
        self.assertIn(self.ROW, gates.RUNNABLE_BASELINE)
        spec = (
            gates.ROOT / ".agents/specs/tenstorrent-qwen35.md"
        ).read_text(encoding="utf-8")
        gate_section = spec.split("\n## Gates\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn("tests/test_qwen35_paged_engine", gate_section)
        self.assertIn("scripts/agent-preflight.sh", gate_section)
        self.assertNotIn("git diff", gate_section)


class KeepquantRunnablePopulationTests(unittest.TestCase):
    """The BACKEND-TENSTORRENT-KEEPQUANT entry is the whole of what #2959's
    staging change did to this checker, so the credit has to be checkable
    rather than plausible: the row must be IN the pinned population, its audit
    verdict must be earned by the record (not merely asserted), and removing
    the entry must break the exact pin -- which is what proves the row was
    pinned because it ENTERED the runnable population and not to quiet a
    gate."""

    ROW = "BACKEND-TENSTORRENT-KEEPQUANT"

    def test_the_row_is_pinned_and_its_verdict_is_runnable(self):
        self.assertIn(self.ROW, gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get(self.ROW), "runnable")

    def test_dropping_the_keepquant_entry_breaks_the_pin(self):
        # MUTATION, in the direction this pin actually moved (#2959): remove
        # the entry and the exact-population equality inside the checker has
        # to go red, because the audit still reports the row as runnable from
        # its spec.
        reduced = set(gates.RUNNABLE_BASELINE) - {self.ROW}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {self.ROW})

    def test_the_keepquant_row_is_credited_for_real_commands(self):
        # A record row can only be credited for the checker that reads the
        # record, so the load-bearing assertion is the last one: the `## Gates`
        # section of the row's own spec must name a command that genuinely
        # fails on a broken tree -- the preflight chain, which reded on this
        # very change before the entry was pinned -- not a command that exits
        # 0 in any tree (`git diff`).
        self.assertIn(self.ROW, gates.RUNNABLE_BASELINE)
        spec = (
            gates.ROOT / ".agents/specs/tenstorrent-keepquant.md"
        ).read_text(encoding="utf-8")
        gate_section = spec.split("\n## Gates\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn("scripts/agent-preflight.sh", gate_section)
        self.assertNotIn("git diff", gate_section)


class Iq4nlRunnablePopulationTests(unittest.TestCase):
    """The QUANT-GGUF-IQ4_NL entry is the whole of what the IQ4_NL ROCm dot
    row changed in this checker, so the credit has to be checkable rather than
    plausible: the row must be IN the pinned population, its audit verdict must
    be earned by the record (not merely asserted), and removing the entry must
    break the exact pin -- which is what proves the row was pinned because it
    ENTERED the runnable population and not to quiet a gate."""

    ROW = "QUANT-GGUF-IQ4_NL"

    def test_the_row_is_pinned_and_its_verdict_is_runnable(self):
        self.assertIn(self.ROW, gates.RUNNABLE_BASELINE)
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get(self.ROW), "runnable")

    def test_dropping_the_iq4nl_entry_breaks_the_pin(self):
        # MUTATION, in the direction this pin actually moved: remove the entry
        # and the exact-population equality inside the checker has to go red,
        # because the audit still reports the row as runnable from its spec.
        reduced = set(gates.RUNNABLE_BASELINE) - {self.ROW}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {self.ROW})

    def test_the_iq4nl_row_is_credited_for_real_commands(self):
        # A record row can only be credited for the checker that reads the
        # record, so the load-bearing assertion is the last one: the `## Gates`
        # section of the row's own spec must name a command that genuinely
        # fails on a broken tree -- the unit suite this row's ROCm arm reded
        # and then greened on -- not a command that exits 0 in any tree
        # (`git diff`).
        self.assertIn(self.ROW, gates.RUNNABLE_BASELINE)
        spec = (
            gates.ROOT / ".agents/specs/quant-gguf-iq4nl.md"
        ).read_text(encoding="utf-8")
        gate_section = spec.split("\n## Gates\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn("ctest --test-dir build", gate_section)
        self.assertNotIn("git diff", gate_section)
