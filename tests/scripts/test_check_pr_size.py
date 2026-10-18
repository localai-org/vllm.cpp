#!/usr/bin/env python3
"""Mutation tests for explicit PR path classes and the checker contracts.

The per-class line budgets were retired 2026-08-10; the tests that pinned them
are gone and `test_no_line_budget_is_enforced_for_any_class` pins their absence."""

from __future__ import annotations

import ast
import datetime as dt
import importlib.util
import os
import re
import shutil
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
# root on sys.path, so provide it before executing the module.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("check_pr_size", ROOT / "scripts/check-pr-size.py")
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def _executed_programs(source: str) -> set[str]:
    """Programs a checker EXECUTES, read from its argument-list literals.

    Derived rather than listed, so a checker that starts calling a new binary
    is caught here instead of in CI as somebody else's failure (#1892).
    """

    programs: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.List) or not node.elts:
            continue
        first = node.elts[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        value = first.value
        if value and "/" not in value and " " not in value and not value.startswith("-"):
            programs.add(value)
    return programs


class CheckerEvidenceMapping(unittest.TestCase):
    # A checker whose evidence path does not exist can never satisfy the
    # governance_checker rule: every change to it fails with "requires semantic
    # mutation evidence in <file that is not there>". That is how
    # check-device-leakage.py became unmodifiable -- its suite predates the
    # test_check_<name> convention, so the derived path missed by one word.
    KNOWN_UNTESTED = {
        # No suite at all, and not wired into ci.yml either. Named here so the
        # gap is visible in a test rather than invisible in a naming rule.
        "scripts/check-dsv4-gguf-namemap.py",
    }

    def test_every_checker_maps_to_an_evidence_file_that_exists(self) -> None:
        root = Path(checker.__file__).resolve().parents[1]
        missing = []
        for path in sorted(root.glob("scripts/check-*.py")):
            rel = f"scripts/{path.name}"
            if rel in self.KNOWN_UNTESTED:
                continue
            evidence = checker.recognized_evidence(rel)
            if not (root / evidence).is_file():
                missing.append(f"{rel} -> {evidence}")
        self.assertEqual(missing, [])


class PathClassification(unittest.TestCase):
    def test_canonical_issue_files_are_project_records(self) -> None:
        for path in (
            ".agents/issues/ARCH-ONE-SURFACE/ISSUE-GH-1195.md",
            ".agents/issues/_owed/ISSUE-GH-2390.md",
            ".agents/issues/_intake/ISSUE-GH-41.md",
            ".agents/issues/ENG-RECORD-CONFLICT-SURFACES/"
            "ISSUE-" "LOCAL-01ARZ3NDEKTSV4RRFFQ69G5FAV.md",
        ):
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), "project_record")

    def test_canonical_issue_file_near_misses_fail_closed(self) -> None:
        for path in (
            ".agents/issues/ARCH-ONE-SURFACE/nested/ISSUE-GH-1195.md",
            ".agents/issues/ARCH-ONE-SURFACE/ISSUE-GH-0.md",
            ".agents/issues/ARCH-ONE-SURFACE/ISSUE-GH-01195.md",
            ".agents/issues/ARCH-ONE-SURFACE/"
            "ISSUE-LOCAL-01ARZ3NDEKTSV4RRFFQ69G5FAI.md",
            ".agents/issues/_intake/"
            "ISSUE-" "LOCAL-01ARZ3NDEKTSV4RRFFQ69G5FAV.md",
            ".agents/issues/ISSUE-GH-1195.md",
            ".agents/issues/_other/ISSUE-GH-1195.md",
            ".agents/issues/ARCH-ONE-SURFACE/notes.md",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_recorded_exl3_artifacts_are_evidence(self) -> None:
        """#3060: inspect every recorded path, including the incomplete leg."""
        runs = {
            "qwen38-27b-exl3-headtohead-20260903": (
                "OURS-A.clientlog", "OURS-B.clientlog",
                "THEIRS-A.clientlog", "THEIRS-B.clientlog",
                "serve_openai-usage.patch",
            ),
            "qwen38-27b-exl3-variadic-20260905": (
                "OURS-r1-c1.clientlog", "OURS-r1-c4.clientlog", "OURS-r1-c8.clientlog",
                "OURS-r2-c1.clientlog", "OURS-r2-c4.clientlog", "OURS-r2-c8.clientlog",
                "PROBE.clientlog",
                "THEIRS-r1-c1.clientlog", "THEIRS-r1-c4.clientlog", "THEIRS-r1-c8.clientlog",
                "THEIRS-r2-c8.clientlog",
            ),
        }
        for run, names in runs.items():
            for name in names:
                path = f"docs/bench-evidence/{run}/{name}"
                with self.subTest(path=path):
                    self.assertEqual(checker.classify_path(path), "evidence")

    def test_recorded_exl3_evidence_near_misses_fail_closed(self) -> None:
        head = "docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903"
        variadic = "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905"
        for path in (
            f"{head}/OURS-C.clientlog",
            f"{head}/PROBE.clientlog",
            f"{variadic}/THEIRS-r3-c1.clientlog",
            f"{variadic}/serve_openai-usage.patch",
            f"{head}/other.patch",
            f"{head}/OURS-A.patch",
            f"{head}/OURS-A.clientlog.bak",
            f"{head}/ours-a.clientlog",
            f"{head}/OURS-A.CLIENTLOG",
            f" {head}/OURS-A.clientlog",
            f"{head}/OURS-A.clientlog ",
            f"{head}/nested/OURS-A.clientlog",
            f"{head}/nested/serve_openai-usage.patch",
            "docs/bench-evidence/OURS-A.clientlog",
            "docs/bench-evidence/serve_openai-usage.patch",
            "docs/bench-evidence/uninspected-run/OURS-A.clientlog",
            "docs/bench-evidence/uninspected-run/serve_openai-usage.patch",
            "docs/other/OURS-A.clientlog",
            "docs/other/serve_openai-usage.patch",
            f"./{head}/OURS-A.clientlog",
            f"{head}//OURS-A.clientlog",
            f"{head}/../qwen38-27b-exl3-headtohead-20260903/OURS-A.clientlog",
            f"{head}/serve_openai-usage.patch/",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_recorded_exl3_evidence_preserves_existing_classes(self) -> None:
        head = "docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903"
        variadic = "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905"
        expected = {
            f"{head}/README.md": "public_document",
            f"{variadic}/corpus-manifest.json": "public_document",
            f"{head}/job-as-run.sh": "evidence",
            f"{head}/client.py": "evidence",
            f"{head}/results.txt": "evidence",
            f"{variadic}/job-as-run.sh": "evidence",
            "benchmarks/variadic/job.sh": "product",
            "benchmarks/variadic/OURS-A.clientlog": "product",
            "benchmarks/variadic/serve_openai-usage.patch": "product",
            "tools/serve_openai.py": "product",
            "src/serve_openai-usage.patch": "product",
            "scripts/check-pr-size.py": "governance_checker",
            "tests/scripts/test_check_pr_size.py": "governance_test",
        }
        for path, path_class in expected.items():
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), path_class)

    def test_state_migration_manifest_archives_are_evidence(self) -> None:
        for path in (
            ".agents/completed/state-migration-manifest.csv",
            ".agents/completed/state-migration-manifest-f921.csv",
            ".agents/completed/state-migration-manifest-release-v1.2.csv",
        ):
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), "evidence")

    def test_both_sides_of_the_issue_index_retirement_classify(self) -> None:
        """#2290 moved the index to `.agents/completed/`, and a diff that spans
        the move names BOTH paths -- the archive as the destination, the old path
        as the rename source.

        Classification is a HARD ERROR, not a skip, so an unclassified side
        aborts pr-size entirely: on the retirement change itself, and on every
        in-flight branch whose range still straddles it.

        A REGRESSION GUARD, not mutation evidence -- it passes at base and at
        head. It is here because the retirement change first "tidied" the old
        path out of the classification set, having assumed the archive needed
        adding; in fact `.agents/completed/` already classifies by prefix, and
        the entry that had to stay was the one removed. pr-size then aborted on
        its own change, and only did so visibly because W8 had just put it in
        preflight. Both sides are pinned so the next reader does not repeat it.
        """
        for path in (
            ".agents/completed/issue-index.md",
            ".agents/issue-index.md",
        ):
            with self.subTest(path=path):
                # classify_path raising ValueError is the failure being pinned.
                self.assertIsInstance(checker.classify_path(path), str)

    def test_a_profiler_csv_under_bench_evidence_is_evidence(self) -> None:
        """#2316: `ncu --csv` writes one, and #2289 landed
        `docs/bench-evidence/laguna-grouped-gemv-ncu-20260829.csv` with no class.

        `classify_path` fails closed, so the whole-tree sweep in this suite went
        RED on `origin/main` itself. It also blocked #2290: that change edits
        this checker, which makes THIS suite its mutation evidence, and the
        evidence contract runs the whole module -- so one unrelated red in it
        fails the pair. RED at base.
        """
        self.assertEqual(
            checker.classify_path(
                "docs/bench-evidence/laguna-grouped-gemv-ncu-20260829.csv"
            ),
            "evidence",
        )

    def test_a_python_probe_in_a_per_run_evidence_dir_is_evidence(self) -> None:
        """#2612: wave GDNDECOMP's instrument is a numpy replica of FLA, so its
        five scripts land as `.py` inside
        `docs/bench-evidence/gdn-chunked-decomposition-20260902/`. The extension
        list carried `txt|log|gz|sh|cu` and not `py`, `classify_path` fails
        closed, and the whole-tree sweep below went RED on the branch carrying
        them -- and would have gone red on `main` on the day they landed, the
        same shape as #1448 and #2316.

        A `.py` is the recipe exactly as a `.sh` is. RED before the `py` arm of
        BENCH_EVIDENCE_RUN.
        """
        for path in (
            "docs/bench-evidence/gdn-chunked-decomposition-20260902/gdn_decomp.py",
            "docs/bench-evidence/gdn-chunked-decomposition-20260902/run_golden.py",
            "docs/bench-evidence/gdn-chunked-decomposition-20260902/check_bf16_helper.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), "evidence")

    def test_a_python_file_outside_a_per_run_evidence_dir_fails_closed(self) -> None:
        """The suffix is not a licence, and this is the arm that keeps a product
        or checker script from taking the evidence class by being spelled `.py`.
        A flat `docs/bench-evidence/*.py` is NOT a per-run directory and does not
        classify either."""
        for path in (
            "docs/some-other-place/whatever.py",
            "docs/bench-evidence/flat-not-a-run-dir.py",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_a_csv_outside_bench_evidence_still_fails_closed(self) -> None:
        """The extension is not a licence. Widening by suffix alone would let a
        csv anywhere in docs/ take the evidence class without review."""
        with self.assertRaises(ValueError):
            checker.classify_path("docs/some-other-place/whatever.csv")

    def test_completed_csv_near_misses_fail_closed(self) -> None:
        for path in (
            ".agents/completed/unrelated.csv",
            ".agents/completed/state-migration-manifests-f921.csv",
            ".agents/completed/state-migration-manifest-.csv",
            ".agents/completed/state-migration-manifest-f921-.csv",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_each_mutable_surface_has_an_explicit_class(self) -> None:
        expected = {
            "CLAUDE.md": "procedure",
            "MANIFESTO.md": "public_document",
            "src/vt/x.cpp": "product",
            "scripts/check-release-binary-contract.py": "product",
            "scripts/check-doc-checkpoint.py": "governance_checker",
            "tests/scripts/test_policy_contract.py": "governance_test",
            ".agents/policy.csv": "policy",
            ".agents/state-index/2026-08-001.csv": "append_only_record",
            ".agents/state-events/2026-08/STATE-20260808T120000-001.md": "append_only_record",
            ".agents/completed/state-migration-manifest.csv": "evidence",
            "docs/STATUS.md": "public_document",
            "website/hugo.toml": "public_document",
            "website/layouts/_default/baseof.html": "public_document",
            "website/assets/css/site.css": "public_document",
            "website/static/fonts/sora-700.woff2": "asset",
            "website/static/logo.svg": "asset",
            ".github/workflows/ci.yml": "ci",
            "src/vt/vulkan/vulkan_spirv.cpp": "generated",
            "release/manifest-v1.schema.json": "configuration",
            "release/release-matrix.json": "configuration",
            "release/release-version.json": "configuration",
            "release/container-matrix.json": "configuration",
            "scripts/env-doc-allowlist.txt": "configuration",
            # The container lane images. `docker/Dockerfile.arm64` already
            # classified through the suffixed pattern, but the unsuffixed
            # multi-lane Dockerfile and its healthcheck did not, and
            # classify_path FAILS CLOSED -- so adding them to the tree made the
            # PR-size gate reject the change outright.
            "docker/Dockerfile": "configuration",
            "docker/Dockerfile.arm64": "configuration",
            "docker/healthcheck.sh": "configuration",
            "tests/scripts/fixtures/release_manifest/v1/cpu-input.json": "asset",
            # Same failure as the Dockerfile entries above, one row later.
            # #840 moved the intake table out of roadmap_v1.md into its own
            # append-only file and never classified it, and classify_path FAILS
            # CLOSED -- so pr-size aborted on every pull request that appends an
            # index row, which under that same policy is nearly all of them
            # (#856). It is a project record for the same reason roadmap_v1.md
            # is: it IS the table roadmap_v1.md used to hold.
            ".agents/roadmap_v1.md": "project_record",
            ".agents/issue-index.md": "project_record",
            ".agents/style/commits.md": "procedure",
            ".agents/style/prose.md": "procedure",
            ".claude/skills/writing-commits-and-prs/SKILL.md": "procedure",
            ".claude/skills/writing-technical-english/SKILL.md": "procedure",
        }
        for path, path_class in expected.items():
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), path_class)

    def test_release_version_classification_is_exact_and_fail_closed(self) -> None:
        """Only the authoritative immutable declaration earns this class.

        A directory-level ``release/`` rule would silently classify future
        mutable release state. Near-miss names therefore remain unknown.
        """
        self.assertEqual(
            checker.classify_path("release/release-version.json"),
            "configuration",
        )
        for path in (
            "release/version.json",
            "release/release-version.local.json",
            "release/channel.json",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_generated_class_does_not_swallow_its_own_sources(self) -> None:
        # The generator and the GLSL it compiles are the REVIEWABLE surface and
        # must keep their own tighter budgets. If either ever classified as
        # `generated`, a shader change could arrive unreviewed behind the blob.
        self.assertEqual(
            checker.classify_path("src/vt/vulkan/shaders/vt_matmul_vec.comp"), "product"
        )
        self.assertEqual(checker.classify_path("scripts/gen-vulkan-spirv.py"), "product")

    def test_generated_files_are_actually_generated_and_gate_verified(self) -> None:
        # The class is only sound while every member is machine-emitted and
        # reproduced by a gate. Assert the self-declaration at the head of each
        # file so a hand-written file cannot be parked here to dodge review.
        root = Path(checker.ROOT)
        self.assertTrue(checker.GENERATED_FILES, "the class must not be empty")
        for rel in checker.GENERATED_FILES:
            with self.subTest(path=rel):
                head = (root / rel).read_text(errors="replace")[:400]
                self.assertIn("GENERATED FILE - DO NOT EDIT BY HAND", head)

    def test_unknown_and_noncanonical_paths_fail_closed(self) -> None:
        for path in (
            "unknown.bin",
            "/scripts/check-doc-checkpoint.py",
            "scripts/../src/x.cpp",
            "scripts//check-doc-checkpoint.py",
            r"scripts\check-doc-checkpoint.py",
            "./scripts/check-doc-checkpoint.py",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_the_model_porting_checklist_is_procedure_and_agents_is_not_blanket(
        self,
    ) -> None:
        """The new per-model checklist classifies, and `.agents/` stays explicit.

        `.agents/porting-a-model.md` (#318) is a task guide like its siblings, so
        it belongs to the procedure class. It is listed by exact path rather than
        by widening a glob over `.agents/`, because AGENTS.md forbids hiding
        mutable files behind a blanket directory exemption.

        Both halves matter. Without the first the gate fails closed on the guide
        -- which is how this was found. Without the second, adding the entry as a
        directory pattern would silently classify every future `.agents/` file,
        including ones nobody reviewed, so the check asserts an unknown sibling
        still fails to classify.
        """
        self.assertEqual(
            checker.classify_path(".agents/porting-a-model.md"), "procedure"
        )
        # Its siblings are unchanged.
        self.assertEqual(checker.classify_path(".agents/porting.md"), "procedure")

        # `.agents/` is NOT a blanket exemption. An unlisted path there does not
        # quietly inherit a class -- the classifier refuses it outright, which is
        # what makes the gate fail closed on every new file rather than only on
        # the ones someone remembered to think about.
        with self.assertRaises(ValueError):
            checker.classify_path(".agents/not-a-real-guide-xyz.md")

    def test_a_measurement_runs_own_artifact_kinds_classify_as_evidence(self) -> None:
        """#2497: a `rc` job's own OUTPUT is evidence, not unclassified paths.

        Distinct from the `py` arm #2629 landed hours earlier, and found
        independently in another directory. That arm covers the RECIPE. These
        two are what the job emitted: `docs/bench-evidence/rocm-strix-llamacpp-
        denominator-20260902/` carries six `clock-leg*.jsonl` and six `leg*.rc`
        with no class, and `classify_path` fails closed, so
        `test_every_tracked_and_current_change_path_is_classified` was RED
        against the WHOLE TREE. Landing that would have reddened `main` itself
        and then refused every later change touching this checker through its
        own evidence contract. It went unseen because this checker is CI-only
        and `agent-preflight.sh` never runs it.

        RED-BEFORE for the `jsonl|rc` arms: both raise `ValueError` against the
        base checker. The `.py` line is a REGRESSION GUARD and is green at base
        as well as at head -- #2629 made it so, and it is kept because these two
        files are the row's own and a later narrowing of that arm would silently
        take them with it.
        """
        run = "docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902"
        for path in (
            f"{run}/clock-leg1.jsonl",     # RED at base: the instrument stream
            f"{run}/leg1.rc",              # RED at base: one leg's exit status
            f"{run}/fold.py",              # green at base since #2629
        ):
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), "evidence")

        # The additions are a closed enumeration over ONE directory shape, not a
        # relaxation. A sibling extension nobody has argued for still fails
        # closed, both new suffixes fail closed outside a per-run evidence
        # directory, and names that already had a class keep it.
        for path in (
            f"{run}/leg1.pickle",
            "docs/bench-evidence/flat-not-a-run-dir.jsonl",
            "docs/some-other-place/whatever.rc",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)
        self.assertEqual(
            checker.classify_path("scripts/check-pr-size.py"), "governance_checker"
        )
        # `.json` stays OUT of the evidence arm: DOC already claims it, and the
        # evidence arm is tested first, so admitting it would reclassify
        # docs/bench-evidence/mxfp4-qwen's golden.
        self.assertEqual(checker.classify_path(f"{run}/leg1.json"), "public_document")

    def test_every_tracked_and_current_change_path_is_classified(self) -> None:
        paths = set(
            subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        )
        paths.update(
            subprocess.check_output(
                ["git", "diff", "--name-only", "origin/main"], cwd=ROOT, text=True
            ).splitlines()
        )
        failures = []
        for path in sorted(paths):
            try:
                checker.classify_path(path)
            except ValueError:
                failures.append(path)
        self.assertEqual(failures, [])

    def test_similar_names_do_not_enter_governance_classes(self) -> None:
        self.assertNotEqual(
            checker.classify_path("scripts/checkpoint-runner.py"),
            "governance_checker",
        )
        self.assertNotEqual(
            checker.recognized_evidence("scripts/check-doc-checkpoint.py"),
            "tests/scripts/test_doc_checkpoint_extra.py",
        )
        self.assertEqual(
            checker.recognized_evidence("scripts/check-agent-record.py"),
            "tests/scripts/test_agent_record.py",
        )


class DeletedCheckers(unittest.TestCase):
    """A checker that is GONE at head owes no mutation evidence.

    The evidence contract asks a checker change to prove a guarantee moved: the
    paired test must fail against the BASE checker and pass against the HEAD one.
    A DELETION has no head checker to run and, when the suite is removed in the
    same change, no test module to import either -- so the contract collapses
    into `ModuleNotFoundError` reported as "HEAD checker/test pair failed",
    which reads as a broken checker rather than an absent one.

    Found by #2290, which deletes `check-issue-index-append-only.py` together
    with its suite. `pr-size` is a REQUIRED check, so a change that retires a
    checker could not land at all. You cannot mutate a guarantee that no longer
    exists; the deletion IS the change, and the review of it is the evidence.
    """

    def test_a_deleted_checker_does_not_demand_evidence(self) -> None:
        changes = [checker.ChangedPath("scripts/check-gone.py", 0, 120)]
        errors = checker.change_errors(changes, deleted={"scripts/check-gone.py"})
        self.assertEqual(errors, [], f"a deletion was asked to prove itself: {errors}")

    def test_a_surviving_checker_still_demands_evidence(self) -> None:
        """The half that must NOT move. Without this the fix above is a mute
        switch: marking every checker change 'deleted' would silence the whole
        contract."""
        changes = [checker.ChangedPath("scripts/check-gone.py", 4, 4)]
        errors = checker.change_errors(changes, deleted=set())
        self.assertTrue(
            any("requires semantic mutation evidence" in e for e in errors),
            f"a live checker change escaped the evidence contract: {errors}",
        )

    def test_a_deleted_checker_whose_suite_survives_still_demands_nothing(self) -> None:
        """Deletion is decided by the CHECKER's absence, not the suite's.

        A change may retire a checker and keep its suite for a while. The
        evidence contract is about the checker, so the checker's absence settles
        it either way.
        """
        changes = [
            checker.ChangedPath("scripts/check-gone.py", 0, 120),
            checker.ChangedPath("tests/scripts/test_check_gone.py", 3, 0),
        ]
        errors = checker.change_errors(changes, deleted={"scripts/check-gone.py"})
        self.assertEqual(errors, [], f"{errors}")


class RetiredSurfaces(unittest.TestCase):
    """Deleted paths keep a class on purpose; the table says so in one place.

    `classify_path` fails closed, and a deleted file still appears in the diff of
    the commit that removes it, so a retired surface with no class reds the very
    change that retires it. The retention was previously spread across three
    live groups and two module-level regexes with the reason written in only one
    of them, and was read as abandoned scaffolding.
    """

    LIVE_GROUPS = (
        "POLICY_FILES",
        "APPEND_ONLY_FILES",
        "PROJECT_RECORD_FILES",
        "PROCEDURE_FILES",
        "GOVERNANCE_SUPPORT_FILES",
        "PRODUCT_CHECKER_FILES",
        "PUBLIC_DOCUMENT_FILES",
        "GENERATED_FILES",
    )

    def test_retired_paths_keep_the_class_they_had_while_live(self) -> None:
        """The budget a historical diff spends must not move under this table."""
        patterns = {
            ".agents/state-index/2026-08-001.csv": "append_only_record",
            ".agents/state-events/2026-08/STATE-20260808T120000-001.md": "append_only_record",
        }
        exact = {
            # Retired by #281: the waiver registry became a commit-message
            # argument recorded in git, so these three left the tree together.
            ".agents/waivers.csv": "policy",
            "scripts/waivers.py": "governance_support",
            "tests/scripts/test_waivers.py": "checker_test",
            ".agents/policy.csv": "policy",
            ".agents/policy-cutover": "policy",
            ".agents/state.md": "project_record",
            ".agents/state.csv": "project_record",
            ".agents/ai-coding-assistants.md": "procedure",
            ".agents/benchmark-protocol.md": "procedure",
            ".agents/directives.md": "procedure",
            ".agents/discipline.md": "procedure",
            ".agents/gates.md": "procedure",
            ".agents/test-porting.md": "procedure",
        }
        for path, path_class in {**exact, **patterns}.items():
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), path_class)
        self.assertEqual(dict(checker.RETIRED_PATHS), exact)

    def test_every_retired_path_is_really_gone_from_the_tree(self) -> None:
        """A live path parked here would take the retired budget and no review.

        If one is ever re-added it must move back to a live group, so this fails
        rather than letting the table quietly govern a live surface.
        """
        tracked = set(
            subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        )
        self.assertEqual(sorted(set(checker.RETIRED_PATHS) & tracked), [])
        still_here = [
            path
            for path in tracked
            for pattern, _ in checker.RETIRED_PATTERNS
            if pattern.fullmatch(path)
        ]
        self.assertEqual(still_here, [])

    def test_the_live_archive_does_not_classify_as_retired(self) -> None:
        """`.agents/completed/state-events/` was MOVED, not deleted: 160 files."""
        archived = ".agents/completed/state-events/2026-08/STATE-20260808T120000-001.md"
        self.assertIsNone(checker.retired_class(archived))
        self.assertEqual(checker.classify_path(archived), "procedure")
        self.assertEqual(
            checker.classify_path(".agents/completed/state-migration-manifest.csv"),
            "evidence",
        )

    def test_retired_and_live_groups_are_disjoint(self) -> None:
        """One home per path: the split that made this look like dead scaffolding."""
        retired = set(checker.RETIRED_PATHS)
        for name in self.LIVE_GROUPS:
            with self.subTest(group=name):
                self.assertEqual(sorted(retired & set(getattr(checker, name))), [])

    def test_a_surface_that_never_EXISTED_fails_closed(self) -> None:
        """`.agents/governance-tasks.csv` was never added and never deleted.

        It sat in POLICY_FILES as a speculative entry, so a file by that name
        could have arrived and spent the policy budget without anyone choosing
        its class. Retirement is for paths git actually removed; this one is
        simply unknown, and unknown fails closed.
        """
        with self.assertRaises(ValueError):
            checker.classify_path(".agents/governance-tasks.csv")
        self.assertNotIn(".agents/governance-tasks.csv", checker.RETIRED_PATHS)


class BudgetEnforcement(unittest.TestCase):
    def change(self, path: str, lines: int) -> checker.ChangedPath:
        return checker.ChangedPath(path, lines, 0)

    def test_no_line_budget_is_enforced_for_any_class(self) -> None:
        """Budgets were retired 2026-08-10; size is a review judgement now.

        RED against the previous checker on both halves: it exported a
        PATH_CLASS_BUDGETS table, and a 100,000-line product change tripped its
        900-line limit. Either surviving would mean the retirement did not land.
        """
        self.assertFalse(hasattr(checker, "PATH_CLASS_BUDGETS"))
        huge = self.change("src/vllm/model_executor/models/qwen3_5.cpp", 100_000)
        self.assertEqual(checker.change_errors([huge]), [])

    def test_retiring_the_budget_did_not_retire_the_other_contracts(self) -> None:
        """The rules that share this checker must still bite.

        Dropping a size gate is not licence to drop classification or
        checker-evidence with it, which is exactly the kind of thing that goes
        unnoticed when a constant is deleted.
        """
        unknown = checker.ChangedPath("no/such/surface.txt", 1, 0)
        self.assertTrue(checker.change_errors([unknown]))
        lone_checker = self.change("scripts/check-pr-size.py", 10)
        self.assertTrue(
            any("mutation evidence" in e for e in checker.change_errors([lone_checker]))
        )

    def test_every_secondary_oracle_file_classifies(self) -> None:
        """One file per oracle must classify (GATE-PR-SIZE-BINARY follow-on, #668).

        RED before the fix on EVERY tracked file under .agents/oracles/: the
        secondary-oracle registry landed with no pattern in the checker, so a
        required check refused any PR that recorded a pin -- which is the one
        thing the registry exists to make cheap. Asserted on the whole tracked
        set rather than a sample, so a ninth oracle added without a class is
        caught here and not in someone's PR.
        """
        tracked = subprocess.run(
            ["git", "ls-files", ".agents/oracles/"],
            capture_output=True, text=True, check=True, cwd=checker.ROOT,
        ).stdout.split()
        self.assertTrue(tracked, "expected tracked .agents/oracles/ files")
        for path in tracked:
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), "procedure")

    def test_oracles_is_a_pattern_not_a_blanket_directory_exemption(self) -> None:
        """The class is earned by shape, not by living under .agents/oracles/.

        AGENTS.md forbids hiding mutable files behind a blanket directory
        exemption, so a non-.md file or a nested path there must still fail
        closed rather than inherit `procedure`.
        """
        for path in (
            ".agents/oracles/pin.txt",
            ".agents/oracles/vllm.json",
            ".agents/oracles/nested/dir.md",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_a_classified_binary_is_accepted(self) -> None:
        """A binary at a classified path is not an error (GATE-PR-SIZE-BINARY, #615).

        RED before the retirement: `change_errors` short-circuited on every
        `lines is None` path, so a captured parity golden could not reach main
        at all and #431 was unmergeable by construction. The classifier was
        always built to give binaries a class -- the `SITE_ASSET` comment says
        so in as many words -- and the guard refused them anyway.
        """
        for path in (
            "tests/parity/goldens/qwen3_greedy_0_6b/our_ids.npy",
            "tests/parity/goldens/qwen35_greedy_0_8b/neartie_gap_mnats.npy",
            "tests/parity/goldens/qwen3_greedy_0_6b/p0_prompt.i32",
            "website/static/fonts/sora-700.woff2",
        ):
            with self.subTest(path=path):
                # Asserted through classify_path rather than a hardcoded class
                # so this stays true if a golden is later reclassified.
                checker.classify_path(path)
                binary = checker.ChangedPath(path, None, None)
                self.assertEqual(checker.change_errors([binary]), [])

    def test_an_unclassified_binary_is_still_refused(self) -> None:
        """Retiring the guard must not turn an unclassified path into a free one.

        This is the rail that keeps the retirement scoped: the protection was
        never "binaries are unreviewable", it was "every path earns a class".
        Green on both sides of the change -- classification runs first -- so it
        is a regression pin, not the evidence for the retirement.
        """
        errors = checker.change_errors([checker.ChangedPath("no/such/surface.png", None, None)])
        self.assertTrue(errors)
        # The message must name the real defect. "Not reviewable as text" told
        # the author to fix something about the file; an unclassified path is
        # something they can actually act on.
        self.assertFalse(
            any("not reviewable as text" in error for error in errors), errors
        )

    def test_checker_change_requires_its_recognized_mutation_test(self) -> None:
        changed = [
            self.change("scripts/check-pr-size.py", 5),
            self.change("tests/scripts/test_unrelated.py", 50),
        ]
        errors = checker.change_errors(changed, evidence_results={})
        self.assertTrue(any("mutation evidence" in error for error in errors), errors)
        changed.append(self.change("tests/scripts/test_check_pr_size.py", 5))
        proof = checker.EvidenceResult(
            checker="scripts/check-pr-size.py",
            test_module="tests.scripts.test_check_pr_size",
            head_tests=1,
            head_passed=True,
            base_tests=1,
            base_failed=True,
        )
        self.assertEqual(
            checker.change_errors(
                changed, evidence_results={"scripts/check-pr-size.py": proof}
            ),
            [],
        )

    def test_every_created_checker_has_closed_bootstrap_evidence(self) -> None:
        expected = {
            "scripts/check-arm-isa-build.py",
            "scripts/check-commit-trailers.py",
            "scripts/check-cpu-isa-build.py",
            "scripts/check-cuda-fat-gencode.py",
            "scripts/check-release-workflow.py",
            "scripts/validate-release-archive.py",
            "scripts/check-pr-size.py",
            "scripts/check-prompt-contract.py",
            "scripts/check-triton-aot-multiarch.py",
            # PR #446 created the native Windows portability checker after the
            # range base, so its own suite must reject a closed disabled form.
            "scripts/check-windows-portability.py",
            # The release-state truth checker was created in the same range and
            # owes the identical closed bootstrap proof.
            "scripts/check-windows-release-state.py",
            # 2026-08-10: the docs-site content guard (#224). A checker created
            # in the same PR has no BASE version to mutate, so it registers the
            # disabled form its own tests must reject.
            "scripts/check-site.py",
            # 2026-08-10: the container-image gates (#170). Both suites load the
            # checker as a module and call into it, so the disabled stub fails
            # every case rather than quietly passing a reduced one.
            "scripts/check-container-matrix.py",
            "scripts/check-container-workflow.py",
            # 2026-08-16: the CUDA arch-gate registration guard (#960). Its suite
            # reaches into the checker's parser, so the disabled stub cannot load.
            "scripts/check-cuda-op-arch-gate.py",
            # 2026-08-18: the symbol-anchor freshness gate (#1143, #1139). Created
            # in the same range, so it has no BASE version to mutate. The disabled
            # stub exits 0 and prints nothing, which fails 20 of its 21 cases --
            # including the clean-tree case, which asserts a checked count at or
            # above the recorded floor and so cannot be satisfied by silence.
            "scripts/check-symbol-anchors.py",
            # 2026-08-20: the conflict-marker gate (#1417). Created in the same
            # range, so it has no BASE version to mutate. The disabled stub
            # exits 0 and prints nothing, which fails 16 of its 21 cases --
            # measured. The five that survive assert only that an ordinary
            # document exits 0, or read no checker at all, so silence satisfies
            # them; every case that reads an exit code of 1 or an examined count
            # goes red.
            "scripts/check-conflict-markers.py",
            # 2026-08-21: the attention-rung gate (#1544). Created in the same
            # range, so it has no BASE version to mutate. Its suite loads the
            # checker as a module at import time and every case then calls into
            # it, so the disabled stub -- which defines none of scan_file,
            # has_marker, drift_sites, stale_allowlist_entries or main -- takes
            # all 31 cases red on AttributeError. Measured, not asserted: the
            # suite has no case that passes without touching the checker.
            "scripts/check-attention-rung-consistency.py",
            # 2026-09-02: the ROCm hardware-dp4a intrinsic gate (ROCM-HW-DP4A).
            # Created in the same range, so it has no BASE version to mutate.
            # Its suite imports the checker as a module and every case calls
            # `check(root=...)`, which the disabled stub does not define, so all
            # 6 cases go red on AttributeError. Measured with the stub in place,
            # not asserted: "Ran 6 tests" then "FAILED (errors=6)", with no case
            # passing on a reduced contract.
            "scripts/check-rocm-dp4a-intrinsic.py",
            # 2026-09-12: the DeepSeek-V4 Vision manifest gate (#2411). Created
            # in the same range, so it has no BASE version to mutate. Its suite
            # loads the checker as a module and every case either calls into it
            # or runs a copy of it over a mutated fixture and reads an exit code
            # of 1 with a counted number of disagreements, so the disabled stub
            # -- which exits 0, prints nothing and defines none of the functions
            # -- takes all 28 cases red. Measured with the stub in place, not
            # asserted: "Ran 28 tests" then "FAILED (failures=19, errors=19)",
            # with no case surviving on a reduced contract.
            "scripts/check-deepseek-v4-vision-manifests.py",
        }
        self.assertEqual(set(checker.CREATION_MUTATIONS), expected)
        for path, mutation in checker.CREATION_MUTATIONS.items():
            with self.subTest(path=path):
                compile(mutation, path, "exec")
                self.assertTrue(checker.recognized_evidence(path).endswith(".py"))

    def test_unchanged_or_mode_only_test_is_not_semantic_mutation_evidence(self) -> None:
        errors = checker.change_errors(
            [
                self.change("scripts/check-pr-size.py", 1),
                self.change("tests/scripts/test_check_pr_size.py", 0),
            ]
        )
        self.assertTrue(any("mutation evidence" in error for error in errors), errors)

    def test_fake_or_comment_only_test_edit_cannot_prove_checker_semantics(self) -> None:
        changes = [
            self.change("scripts/check-pr-size.py", 1),
            self.change("tests/scripts/test_check_pr_size.py", 1),
        ]
        fake = checker.EvidenceResult(
            checker="scripts/check-pr-size.py",
            test_module="tests.scripts.test_check_pr_size",
            head_tests=1,
            head_passed=True,
            base_tests=1,
            base_failed=False,
        )
        errors = checker.change_errors(
            changes, evidence_results={"scripts/check-pr-size.py": fake}
        )
        self.assertTrue(any("BASE checker stayed green" in error for error in errors), errors)

    def test_unexecuted_test_fails_closed(self) -> None:
        changes = [
            self.change("scripts/check-pr-size.py", 1),
            self.change("tests/scripts/test_check_pr_size.py", 1),
        ]
        for count in (0,):
            with self.subTest(count=count):
                proof = checker.EvidenceResult(
                    checker="scripts/check-pr-size.py",
                    test_module="tests.scripts.test_check_pr_size",
                    head_tests=count,
                    head_passed=True,
                    base_tests=1,
                    base_failed=True,
                )
                self.assertTrue(
                    checker.change_errors(
                        changes,
                        evidence_results={"scripts/check-pr-size.py": proof},
                    )
                )

    def test_evidence_tools_do_not_leak_the_ambient_path(self) -> None:
        """CMake and Ninja are private copies, not ambient PATH leakage."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ambient = root / "ambient"
            ambient.mkdir()
            empty_system_path = root / "empty-system-path"
            empty_system_path.mkdir()
            for name in ("cmake", "ninja", "pwsh", "ambient-secret"):
                executable = ambient / name
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)
            with mock.patch.dict(
                os.environ, {"PATH": str(ambient)}, clear=True
            ), mock.patch.object(
                checker.os, "defpath", str(empty_system_path)
            ):
                tools = checker._prepare_evidence_tools(
                    root, "tests.scripts.test_check_windows_portability"
                )
                env = checker._sanitized_env(root, tools)
                entries = env["PATH"].split(os.pathsep)
                self.assertNotIn(str(ambient), entries)
                for name in ("cmake", "ninja", "pwsh"):
                    with self.subTest(name=name):
                        self.assertEqual(
                            shutil.which(name, path=env["PATH"]),
                            str(tools / name),
                        )
                self.assertIsNone(
                    shutil.which("ambient-secret", path=env["PATH"])
                )

    def test_registration_evidence_can_reach_every_program_it_executes(self) -> None:
        """#1892: the harness must be able to RUN the module it judges.

        `scripts/check-test-registration.py` starts real programs -- it
        configures CMake, queries CTest, and the suite drives the `Ninja
        Multi-Config` generator. The module was absent from
        `EVIDENCE_REQUIRED_TOOLS`, so the harness gave it an empty private
        tools directory and it died with `FileNotFoundError: 'cmake'` on the CI
        runner, 26 errors charged to whichever checker was under change.
        Declaring `cmake` alone then moved the same failure to `ctest`.

        So the expectation is DERIVED from the checker's own source rather than
        transcribed here: every program it executes must resolve under the
        sanitized environment, either from the system default path or from the
        private tools directory. A checker that starts using a new program
        reds this test instead of reaching CI as somebody else's defect.
        """

        module = "tests.scripts.test_check_test_registration"
        checker_source = (ROOT / "scripts/check-test-registration.py").read_text(
            encoding="utf-8"
        )
        programs = _executed_programs(checker_source)
        # Non-vacuous: a parse that found nothing would pass silently.
        self.assertIn("cmake", programs)
        self.assertIn("ctest", programs)
        # The suite names the generator, which needs the binary behind it.
        self.assertIn(
            "Ninja Multi-Config",
            (ROOT / "tests/scripts/test_check_test_registration.py").read_text(
                encoding="utf-8"
            ),
        )
        programs.add("ninja")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ambient = root / "ambient"
            ambient.mkdir()
            # Stands in for the real `/bin:/usr/bin`, holding only what a bare
            # system provides. Mocked rather than read, so the outcome does not
            # depend on where this host installed cmake.
            system_path = root / "system-path"
            system_path.mkdir()
            for name in ("bash", "python3"):
                executable = system_path / name
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)
            for name in sorted(programs | {"ambient-secret"}):
                executable = ambient / name
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)
            with mock.patch.dict(
                os.environ, {"PATH": str(ambient)}, clear=True
            ), mock.patch.object(checker.os, "defpath", str(system_path)):
                tools = checker._prepare_evidence_tools(root, module)
                env = checker._sanitized_env(root, tools)
                for name in sorted(programs):
                    with self.subTest(program=name):
                        self.assertIsNotNone(
                            shutil.which(name, path=env["PATH"]),
                            f"{module} executes {name}, which the sanitized "
                            "environment cannot reach",
                        )
                self.assertIsNone(shutil.which("ambient-secret", path=env["PATH"]))

    def test_portability_evidence_fails_closed_for_each_missing_tool(self) -> None:
        module = "tests.scripts.test_check_windows_portability"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for missing in ("cmake", "ninja", "pwsh"):
                with self.subTest(missing=missing):
                    container = root / f"container-{missing}"
                    container.mkdir()
                    available = root / f"available-{missing}"
                    available.mkdir()
                    for name in ({"cmake", "ninja", "pwsh"} - {missing}):
                        executable = available / name
                        executable.write_text(
                            "#!/bin/sh\nexit 0\n", encoding="utf-8"
                        )
                        executable.chmod(0o755)
                    with mock.patch.dict(
                        os.environ, {"PATH": str(available)}, clear=True
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"semantic evidence requires executable {missing}",
                        ):
                            checker._prepare_evidence_tools(container, module)

    def test_private_cmake_retains_its_installed_module_tree(self) -> None:
        module = "tests.scripts.test_check_windows_portability"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installation = root / "installation"
            binaries = installation / "bin"
            modules = installation / "share/cmake/Modules"
            binaries.mkdir(parents=True)
            modules.mkdir(parents=True)
            cmake = binaries / "cmake"
            cmake.write_text(
                "#!/usr/bin/python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "root = Path(__file__).resolve().parent.parent\n"
                "if not (root / 'share/cmake/Modules').is_dir():\n"
                "    print('Could not find CMAKE_ROOT', file=sys.stderr)\n"
                "    raise SystemExit(1)\n"
                "print('Build files have been written')\n",
                encoding="utf-8",
            )
            cmake.chmod(0o755)
            ninja = binaries / "ninja"
            ninja.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            ninja.chmod(0o755)
            pwsh = binaries / "pwsh"
            pwsh.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            pwsh.chmod(0o755)
            empty_system_path = root / "empty-system-path"
            empty_system_path.mkdir()
            with mock.patch.dict(
                os.environ, {"PATH": str(binaries)}, clear=True
            ), mock.patch.object(
                checker.os, "defpath", str(empty_system_path)
            ):
                tools = checker._prepare_evidence_tools(root, module)
                result = subprocess.run(
                    [str(tools / "cmake")],
                    env=checker._sanitized_env(root, tools),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn("Build files have been written", result.stdout)

    def test_arbitrary_test_filename_cannot_claim_mutation_evidence(self) -> None:
        errors = checker.change_errors(
            [
                self.change("scripts/check-doc-checkpoint.py", 1),
                self.change("tests/scripts/test_check_doc_checkpoint.py", 1),
            ]
        )
        self.assertTrue(any("test_doc_checkpoint.py" in error for error in errors), errors)

    def test_agent_record_checker_uses_its_existing_mutation_suite(self) -> None:
        self.assertEqual(
            checker.recognized_evidence("scripts/check-agent-record.py"),
            "tests/scripts/test_agent_record.py",
        )



    def test_numstat_parser_rejects_malformed_negative_and_duplicate_paths(self) -> None:
        for text in (
            "1\t2\n",
            "-1\t2\tsrc/x.cpp\n",
            "1\t2\tsrc/x.cpp\n3\t4\tsrc/x.cpp\n",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    checker.parse_numstat(text)

    def test_real_git_rename_is_delete_plus_add_with_no_rename_syntax(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "old.cpp").write_text("line\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", str(repo), "mv", "src/old.cpp", "src/new.cpp"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "rename"], check=True)
            head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            changes = checker.changed_paths(base, head, repo=repo)
            self.assertEqual(
                [(change.path, change.added, change.removed) for change in changes],
                [("src/new.cpp", 1, 0), ("src/old.cpp", 0, 1)],
            )
            self.assertFalse(any("{" in change.path or "=>" in change.path for change in changes))

    def test_changed_paths_uses_the_merge_base_when_the_base_branch_moved(self) -> None:
        """A PR diff is merge_base..head, not base_tip..head (#773).

        RED before GATE-FORK-ANCESTRY: CI passes `pull_request.base.sha`, the
        TIP of the base branch, which stops being an ancestor of head the moment
        main advances -- so `changed_paths` raised and classification never ran
        on any fork PR. Worse than strict: two-dot diffing a moved main renders
        MAIN's own commits as reversions inside the contributor's diff, so paths
        they never touched get classified and charged to them. This asserts both
        halves: the PR's path is present, main's is absent.
        """
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            repo = Path(directory)
            run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            run("config", "user.name", "Test")
            run("config", "user.email", "test@example.com")
            (repo / "src").mkdir()
            (repo / "src" / "root.cpp").write_text("root\n", encoding="utf-8")
            run("add", ".")
            run("commit", "-qm", "root")
            root = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()

            # The contributor's branch, cut from root.
            run("checkout", "-q", "-b", "pr", root)
            (repo / "src" / "from_pr.cpp").write_text("pr\n", encoding="utf-8")
            run("add", ".")
            run("commit", "-qm", "pr work")
            head = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()

            # main moves on afterwards -- the ordinary case, not an edge case.
            run("checkout", "-q", "-B", "main", root)
            (repo / "src" / "from_main.cpp").write_text("main\n", encoding="utf-8")
            run("add", ".")
            run("commit", "-qm", "mainline work")
            moved_main = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertNotEqual(moved_main, root)

            paths = {c.path for c in checker.changed_paths(moved_main, head, repo=repo)}
            self.assertIn("src/from_pr.cpp", paths)
            self.assertNotIn(
                "src/from_main.cpp",
                paths,
                "main's own commit must not appear in the contributor's diff",
            )

    def test_missing_and_nonancestor_objects_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "base.cpp").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--orphan", "side"], check=True)
            subprocess.run(["git", "-C", str(repo), "rm", "-qrf", "."], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "side.cpp").write_text("side\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "side"], check=True)
            side = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            with self.assertRaises(ValueError):
                checker.changed_paths("missing", side, repo=repo)
            # Still RAISES -- the fail-closed half of the old non-ancestor rule
            # is deliberately preserved (#773). `side` here is an ORPHAN branch,
            # so there is no merge base and therefore no range to compute. Only
            # the message changed: what used to be reported as "not an ancestor"
            # is now named for what it actually is.
            with self.assertRaisesRegex(ValueError, "no merge base"):
                checker.changed_paths(base, side, repo=repo)

    def test_production_pr_classifier_covers_every_governed_path_class(self) -> None:
        governed = (
            "src/x.cpp",
            ".agents/policy.csv",
            "scripts/check-policy.py",
            "tests/scripts/test_policy_contract.py",
            ".agents/workflow.md",
            ".agents/state-index/2026-08-001.csv",
            ".agents/NOW.md",
            "docs/STATUS.md",
            ".github/workflows/ci.yml",
            "scripts/agent-role.py",
            ".env.example",
            "release/manifest-v1.schema.json",
        )
        for path in governed:
            with self.subTest(path=path):
                self.assertTrue(checker.requires_reviewed_pr(path))

    def test_role_discipline_loader_is_removed(self) -> None:
        """check-role-discipline.py was deleted; its loader is gone too."""
        self.assertFalse(
            hasattr(checker, "load_role_discipline"),
            "load_role_discipline should be removed with check-role-discipline.py",
        )

    def test_role_discipline_evidence_override_is_removed(self) -> None:
        """No evidence override for the deleted checker."""
        self.assertNotIn(
            "scripts/check-role-discipline.py",
            checker.CHECKER_EVIDENCE_OVERRIDES,
        )


class PerClaimPathClass(unittest.TestCase):
    """`.agents/claims/CLAIM-*.md` is a classified path (#364).

    One file per active claim, added because the claims TABLE in
    coordination.md is insert-at-one-anchor and was the largest single conflict
    source measured -- 8 of 16 conflicting open PRs. classify_path FAILS CLOSED
    on an unknown path, so without this the directory cannot be added at all.
    """

    def test_a_claim_file_is_classified(self) -> None:
        self.assertEqual(
            checker.classify_path(".agents/claims/CLAIM-EXAMPLE.md"),
            checker.classify_path(".agents/specs/example.md"),
            "a per-claim file is the same class as the per-row spec it mirrors",
        )

    def test_the_directory_readme_is_classified(self) -> None:
        checker.classify_path(".agents/claims/README.md")

    def test_dropping_the_pattern_fails_closed(self) -> None:
        """MUTATION: without the CLAIM pattern the path is unclassified.

        Proves the added clause is load-bearing rather than shadowed by some
        broader rule that already accepted the path.
        """
        with mock.patch.object(checker, "CLAIM", re.compile(r"(?!)")):
            with self.assertRaises(ValueError):
                checker.classify_path(".agents/claims/CLAIM-EXAMPLE.md")

    def test_a_non_claim_file_in_the_directory_is_still_rejected(self) -> None:
        """The pattern must not become a blanket exemption for the directory.

        AGENTS.md forbids hiding mutable files behind a directory exemption, so
        a non-markdown path here must still fail closed.
        """
        with self.assertRaises(ValueError):
            checker.classify_path(".agents/claims/CLAIM-EXAMPLE.sh")


if __name__ == "__main__":
    unittest.main()
