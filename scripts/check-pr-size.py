#!/usr/bin/env python3
"""Enforce explicit path classification and the checker-evidence contract.

The per-class LINE BUDGETS this file used to enforce were retired 2026-08-10,
and the fail-closed BINARY GUARD was retired 2026-08-13; see the notes where
each stood. What remains: every changed path must classify explicitly, a
governance-checker change must carry executable mutation evidence, and product
paths must arrive on a PR. Nothing here measures the size of a diff."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath



ROOT = Path(__file__).resolve().parents[1]

PATH_CLASSES = frozenset(
    {
        "product",
        "governance_checker",
        "governance_test",
        "governance_support",
        "policy",
        "procedure",
        "append_only_record",
        "project_record",
        "public_document",
        "design",
        "ci",
        "configuration",
        "asset",
        "evidence",
        "vendored_dependency",
        "generated",
    }
)
# NO LINE BUDGET. The per-class budgets that used to live here were retired on
# 2026-08-10 by developer decision, on measured grounds: over the last 22 merged
# PRs the `product` budget of 900 lines was exceeded by 9 of them (41%), and
# tests were 33-57% of the diff in every large one. A gate that fires on four
# changes in ten is not a budget, it is noise that teaches people to waive it,
# and charging RED-first mutation tests against the same allowance as kernel
# code penalised exactly the discipline the rest of this document demands.
# Reviewability is now a review judgement, not an arithmetic one.
#
# Everything else this checker enforces is unchanged and is NOT a size rule:
# explicit path classification (no blanket directory exemptions), the
# checker-change mutation-evidence contract, and the role checks that keep
# product paths on a PR.

# Machine-generated artifacts, each of which MUST be (a) emitted by a tracked
# generator in this repository, (b) reproduced byte-for-byte by a gate that runs
# in CI, and (c) marked "GENERATED FILE - DO NOT EDIT BY HAND" at its head.
# Adding a path here without all three is how this class would become a hole.
GENERATED_FILES = frozenset(
    {
        # scripts/gen-vulkan-spirv.py, from src/vt/vulkan/shaders/*.comp.
        # Reproduced by `gen-vulkan-spirv.py --check` in the vulkan-spirv-freshness
        # CI job; the GLSL it compiles stays `product` and is what review reads.
        "src/vt/vulkan/vulkan_spirv.cpp",
    }
)

POLICY_FILES = frozenset(
    {
    }
)
APPEND_ONLY_FILES = frozenset(
    {
        ".agents/benchmark-record.md",
        ".agents/parity-ledger.md",
    }
)
PROJECT_RECORD_FILES = frozenset(
    {
        ".agents/NOW.md",
        ".agents/coordination.md",
        ".agents/roadmap_v1.md",
        # The intake surface moved out of roadmap_v1.md into its own append-only
        # file (#840) and was never classified here. Classification is a hard
        # error by design, so pr-size aborted on EVERY pull request that appends
        # an index row, which under the same policy is nearly all of them (#856).
        ".agents/issue-index.md",
        ".agents/porting-inventory.md",
        ".agents/engine-matrix.md",
        ".agents/feature-matrix.md",
        ".agents/model-matrix.md",
        ".agents/quantization-matrix.md",
        ".agents/kernel-matrix.md",
        ".agents/backend-matrix.md",
        ".agents/sglang-matrix.md",
    }
)
PROCEDURE_FILES = frozenset(
    {
        "AGENTS.md",
        # A tracked SYMLINK to AGENTS.md, for tools that look for CLAUDE.md. It is
        # the same procedure text and shares its budget; without this the checker
        # failed closed on every change that touched it.
        "CLAUDE.md",
        ".agents/workflow.md",
        ".agents/verification.md",
        # The "nothing lands dead" task guide (#888, `8f49ac3be`). Never
        # classified, and classify_path FAILS CLOSED, so pr-size refused every
        # PR that touched it and this file's own suite has been red on main ever
        # since -- silently, because that suite runs in no CI job and only the
        # next checker change loads it (#989). Third instance of the same class
        # after #856 and #668; listed explicitly rather than letting .agents/
        # become a blanket exemption.
        ".agents/reachability.md",
        ".agents/porting.md",
        # The per-model coverage checklist that porting.md points at (#318). Same
        # procedure class as its sibling guides; listed explicitly rather than
        # letting .agents/ become a blanket exemption.
        ".agents/porting-a-model.md",
        ".agents/benchmarking.md",
        ".agents/bugfixing.md",
        # The writing guides AGENTS.md delegates to, and the skill routes that
        # point at them (#827). Same procedure class as their sibling guides.
        # They were never classified, so classify_path FAILED CLOSED and this
        # test has been red on main ever since that row landed (#856).
        ".agents/style/commits.md",
        ".agents/style/prose.md",
        ".claude/skills/writing-commits-and-prs/SKILL.md",
        ".claude/skills/writing-technical-english/SKILL.md",
        ".agents/prompts/implementer.md",
        ".agents/prompts/operator.md",
        ".agents/prompts/reviewer.md",
        ".agents/backends.md",
        ".agents/developer-preferences.example.md",
        ".agents/environment.md",
        ".agents/mission.md",
        ".agents/parity-lever-protocol.md",
        ".agents/upstream-sync.md",
        ".agents/vllm-v1-v2.md",
    }
)
GOVERNANCE_SUPPORT_FILES = frozenset(
    {
        "scripts/agent-role.py",
        "scripts/claim-view.py",
        "scripts/ready-for-helper.py",
        "scripts/agent-preflight.sh",
    }
)
PRODUCT_CHECKER_FILES = frozenset({"scripts/check-release-binary-contract.py"})
PUBLIC_DOCUMENT_FILES = frozenset(
    {
        "README.md",
        "CONTRIBUTING.md",
        # Landed by a9a8581d and never classified, so the checker failed closed on
        # it the same way it did on CLAUDE.md.
        "MANIFESTO.md",
        "docs/STATUS.md",
        "docs/BENCHMARKS.md",
        "docs/FEATURES.md",
        "docs/USAGE.md",
    }
)
CHECKER = re.compile(r"scripts/check-[a-z0-9]+(?:-[a-z0-9]+)*\.(?:py|sh)\Z")
CHECKER_TEST = re.compile(r"tests/scripts/test_[a-z0-9]+(?:_[a-z0-9]+)*\.py\Z")
CI = re.compile(r"\.github/(?:workflows/[A-Za-z0-9_.-]+\.ya?ml|dependabot\.yml|pull_request_template\.md)\Z")
DESIGN = re.compile(r"docs/superpowers/specs/[0-9]{4}-[0-9]{2}-[0-9]{2}-[a-z0-9-]+\.md\Z")
DOC = re.compile(r"docs/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:md|png|svg|json)\Z")
# The published documentation site. Its layouts, CSS, config and assets are
# prose and presentation for a PUBLIC surface, reviewed the way the documents
# themselves are -- not product code, and not CI (the workflow that publishes it
# keeps its own `ci` class). One class for the whole directory on purpose:
# splitting layouts from config would let a large redesign hide half its diff in
# the cheaper bucket, which is the blanket-exemption failure AGENTS.md names.
SITE = re.compile(r"website/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
# `website/static/` is the site's artwork and fonts -- logos, a favicon, woff2.
# Not prose, and the binaries among them have no reviewable line budget at all,
# so they take the `asset` class the same way any other shipped artwork does.
# Kept as a separate pattern rather than an extension list: what makes these
# assets is WHERE they live, and static/ is Hugo's name for exactly that.
SITE_ASSET = re.compile(r"website/static/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
SPEC = re.compile(r"\.agents/specs/[A-Za-z0-9_.-]+\.md\Z")
SPEC_EVIDENCE = re.compile(r"\.agents/specs/[A-Za-z0-9_.-]+\.(?:patch|json|log)\Z")
COMPLETED = re.compile(r"\.agents/completed/[A-Za-z0-9_.-]+\.md\Z")
# One file per active claim (ENG-RECORD-CONFLICT-SURFACES, #364). The claims
# TABLE in coordination.md was insert-at-one-anchor, so every concurrent claim
# collided there -- 8 of the 16 conflicting open PRs at origin/main d928e2c3,
# including six from ONE author's sequential ROCm stack whose only conflict was
# this. A claim in its own file has one writer and cannot collide. Classified
# with the other per-row records it now resembles.
CLAIM = re.compile(r"\.agents/claims/[A-Za-z0-9_.-]+\.md\Z")
ISSUE_RECORD = re.compile(
    r"\.agents/issues/(?:(?:[A-Z0-9][A-Za-z0-9_.-]*|_owed)/"
    r"(?:ISSUE-GH-[1-9][0-9]*|ISSUE-LOCAL-[0-7][0-9A-HJKMNP-TV-Z]{25})"
    r"|_intake/ISSUE-GH-[1-9][0-9]*)\.md\Z"
)

# One file per secondary oracle (AGENTS.md, "When vLLM has no implementation").
# Same shape and therefore the same class as SPEC and CLAIM: a per-key record
# globbed for reading, deliberately NOT a shared table every change must write.
# Absent until #668 -- the registry landed with no pattern here, so every one of
# its files was unclassified and a required check refused any PR touching a pin.
ORACLE = re.compile(r"\.agents/oracles/[A-Za-z0-9_.-]+\.md\Z")
# Retired state evidence, moved wholesale under completed/ when history became
# git. It is archived evidence, classified like every other completed record.
COMPLETED_STATE_EVENT = re.compile(
    r"\.agents/completed/state-events/\d{4}-\d{2}/STATE-[A-Za-z0-9-]+\.md\Z"
)
SYNC_RECORD = re.compile(r"\.agents/sync/[A-Za-z0-9_.-]+\.md\Z")
HOOK = re.compile(r"\.githooks/(?:README\.md|[A-Za-z0-9_.-]+)\Z")
# `csv` joined the list 2026-08-29 (#2316): `ncu --csv` writes one, and
# `docs/bench-evidence/laguna-grouped-gemv-ncu-20260829.csv` (#2289) landed
# on `main` with no class at all. `classify_path` FAILS CLOSED, so the
# whole-tree sweep in this checker's suite went red on `main` itself. A
# profiler export is evidence exactly as a `.log` is.
BENCH_EVIDENCE = re.compile(r"(?:benchmarks/(?:demo|media)|docs/bench-evidence)/[A-Za-z0-9_.-]+\.(?:json|png|gif|mp4|log|csv)\Z")
# A PER-RUN evidence directory: docs/bench-evidence/<run-id>/<file> (#1448).
# AGENTS.md requires the exact build and run recipe beside a measurement, so one
# run arrives as a dated directory of logs, dumps and the scripts that produced
# them, not as one flat file. BENCH_EVIDENCE above matches exactly ONE path
# segment, so all ten files of the first such directory
# (`gdn-replayssm-w0-20260818`, landed 2026-08-18) were unclassified.
# `classify_path` FAILS CLOSED, and the sweep in
# tests/scripts/test_check_pr_size.py classifies every TRACKED path, so that
# suite went red on `main` and every PR touching this checker was refused
# through its own evidence contract. Fourth instance of the class after #856,
# #668 and #989, and repaired the same way: name the surface, do not widen a
# rule.
#
# The extension list is EXACTLY the union of what the tracked per-run
# directories carry, and it deliberately omits `.md` and `.json`. Those two already classify as
# public_document through DOC below, and the evidence arm is tested FIRST, so
# admitting them here would silently RECLASSIFY
# docs/bench-evidence/mxfp4-qwen/*.md and its golden .json. Preserving the class
# of a path that already had one matters more than making the directory
# uniform.
#
# `.sh` and `.cu` are evidence, not product: they are the recipe that produced
# the number. Nothing builds them, nothing installs them, and no entry point
# reaches them.
#
# `.py` joined them 2026-09-03 (#2629). A numeric study whose instrument is a
# numpy replica arrives as `.py` for exactly the reason a profiler run arrives
# as `.sh`: it IS the recipe. `docs/bench-evidence/gdn-chunked-decomposition-
# 20260902/` landed SIX of them, and `classify_path` FAILS CLOSED, so the
# whole-tree sweep in tests/scripts/test_check_pr_size.py went red on the branch
# carrying them and would have gone red on `main` the moment they landed. At
# least the SEVENTH instance of the class -- #856, #668, #989, #1448, then
# `8496d93dd` (csv, #2316) and `6d1335568` (AGENT_RUN_SCRIPT, #2609), the last
# of those one day earlier -- and repaired the same way: name the surface, do
# not widen a rule.
#
# THE `.sh`/`.cu` PREMISE ABOVE IS TOO STRONG AND THE TREE FALSIFIES IT. Two
# tracked files DO read a path under docs/bench-evidence/ by name:
# scripts/dgx-online-serving.sh:1203 passes
# docs/bench-evidence/mxfp4-qwen/golden_marlin_w4a16.json to
# tools/bench/mxfp4_smoke_gate.py as a gate input, and
# tests/scripts/test_cpu_x86_llamacpp_floor.py:26 reads
# docs/bench-evidence/cpu-x86-llamacpp-20260811.md. So "nothing outside docs/
# references this directory" is false as a universal claim and must not be
# repeated.
#
# What was checked for THIS arm, and what holds: NO `.py` under
# docs/bench-evidence/ is read, imported, built, installed or executed by
# anything outside docs/. The only outside mentions of one are the literal path
# strings in this checker's own test, which classify them rather than run them.
# That is the narrow claim the `py` arm rests on. A `.py` OUTSIDE a per-run
# evidence directory still fails closed, which is what keeps this from becoming
# a suffix licence.
#
# `.jsonl` and `.rc` joined the same day (#2497), from a different directory and
# for a different reason than `.py`. They are not recipes; they are the job's
# own OUTPUT, which the `.py` arm above does not cover:
#
#   `.jsonl` the raw instrument stream, one JSON object per sample. It is the
#            population a folded clock figure is derived FROM, which is what a
#            `.log` already is here, so it takes the same class. `.json` stays
#            deliberately absent for the reason given above, and `.jsonl` does
#            not match DOC.
#   `.rc`    one process's recorded exit status. A leg's `rc` is what decides
#            whether its figure may be used at all, so it is evidence in the
#            strictest sense: two bytes nothing can regenerate.
#
# Twelve paths of `rocm-strix-llamacpp-denominator-20260902` were unclassified
# without them -- six `clock-leg*.jsonl` and six `leg*.rc` -- and two more,
# `fold.py` and `amd_clock_sample.py`, were unclassified until #2629 landed the
# `py` arm above hours earlier. Same failure, same day, two directories, found
# independently.
#
# The narrow claim above is re-checked for these two and holds the same way:
# nothing outside docs/ reads, imports, builds, installs or executes a `.jsonl`
# or `.rc` under docs/bench-evidence/. It is NOT re-stated as the universal
# claim the paragraph above corrects. Outside a per-run evidence directory both
# still fail closed, which is what keeps this a named surface rather than a
# suffix licence.
BENCH_EVIDENCE_RUN = re.compile(
    r"docs/bench-evidence/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
    r"\.(?:txt|log|gz|sh|cu|py|jsonl|rc)\Z"
)
# #3060: exact client stdout and the harness patch archived with two EXL3 runs.
# The patch records an external server-wrapper adaptation, and the live
# variadic recipe applies its staged copy. Keep this admission exact so an
# uninspected log or patch cannot inherit the evidence class.
# Source/history: .agents/specs/gate-pr-size-bench-evidence.md.
RECORDED_BENCH_EVIDENCE = frozenset(
    {
        "docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/OURS-A.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/OURS-B.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/THEIRS-A.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/THEIRS-B.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/serve_openai-usage.patch",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r1-c1.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r1-c4.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r1-c8.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r2-c1.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r2-c4.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r2-c8.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r1-c16.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r1-c32.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r2-c16.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/OURS-r2-c32.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/PROBE.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r1-c1.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r1-c4.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r1-c8.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r1-c16.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r1-c32.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r2-c1.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r2-c4.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r2-c8.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r2-c16.clientlog",
        "docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/THEIRS-r2-c32.clientlog",
    }
)
# #2609. The lease RECIPES: the exact script a `rc` job ran to produce a number
# a spec then cites. Same class and same reasoning as BENCH_EVIDENCE_RUN's `.sh`
# and `.cu` above -- verified, not assumed: nothing under CMakeLists.txt,
# .github/, scripts/, src/ or include/ references `.agents/scripts/`, and the
# only referents are spec files. Nothing builds them, nothing installs them, and
# no entry point reaches them.
#
# This directory had NO class at all, and `classify_path` fails closed, so four
# already-tracked files under it left
# `test_every_tracked_and_current_change_path_is_classified` RED on `main` and
# errored any change that touched one. It went unseen because check-pr-size.py
# is CI-only -- `agent-preflight.sh` never runs it.
AGENT_RUN_SCRIPT = re.compile(
    r"\.agents/scripts/[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?"
    r"\.(?:sh|py|hip|cu)\Z"
)
STATE_MIGRATION_MANIFEST = ".agents/completed/state-migration-manifest.csv"
STATE_MIGRATION_MANIFEST_ARCHIVE = re.compile(
    r"\.agents/completed/state-migration-manifest-"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?\.csv\Z"
)
ASSET = re.compile(r"assets/[A-Za-z0-9_.-]+\.(?:png|svg)\Z")
RELEASE_MANIFEST_FIXTURE = re.compile(
    r"tests/scripts/fixtures/release_manifest/v[0-9]+/[a-z0-9-]+\.json\Z"
)

# RETIRED SURFACES -- deleted from the tree, still classified ON PURPOSE.
#
# `classify_path` fails closed on an unknown path, and a deleted file still
# appears in the diff of the commit that removes it, so a surface with no class
# reds the very change that retires it and every later range spanning that
# commit. That is why these stay.
#
# They live HERE, in one table with the reason written once, because the same
# retention used to be spread through POLICY_FILES, PROJECT_RECORD_FILES,
# PROCEDURE_FILES and two module-level regexes with the reason written in only
# ONE of them -- which reads as abandoned scaffolding rather than a deliberate
# fail-closed guard, and was mistaken for exactly that. A path may appear here or
# in a live group, never both.
#
# Each entry keeps the class its path resolved to while it was live, so the
# review budget a historical diff spends does not move.
RETIRED_PATHS = {
    # `policy: retire the waiver registry` (#281): a registry of exceptions is a
    # state log, so an exception now argues for itself in the commit that needs
    # it and `git log --grep` is the record. Same reasoning as policy.csv below.
    ".agents/waivers.csv": "policy",
    "scripts/waivers.py": "governance_support",
    "tests/scripts/test_waivers.py": "checker_test",
    # `policy: the code is the state, git is the history` (0f3e44ee): AGENTS.md
    # became the single normative surface and git became the history.
    ".agents/policy.csv": "policy",
    ".agents/policy-cutover": "policy",
    ".agents/state.md": "project_record",
    ".agents/state.csv": "project_record",
    # `policy: optimize the agent protocol for strict, executable compliance`
    # (1a021b1b, #128): folded into workflow/verification/porting.md.
    ".agents/ai-coding-assistants.md": "procedure",
    ".agents/benchmark-protocol.md": "procedure",
    ".agents/directives.md": "procedure",
    ".agents/discipline.md": "procedure",
    ".agents/gates.md": "procedure",
    ".agents/test-porting.md": "procedure",
}
RETIRED_PATTERNS = (
    # The CSV index and the per-event files of the retired state record, both
    # removed by 0f3e44ee. Their ARCHIVE under `.agents/completed/state-events/`
    # is LIVE evidence -- 160 files, moved verbatim rather than deleted -- and
    # classifies through COMPLETED_STATE_EVENT, not here.
    (re.compile(r"\.agents/state-index/\d{4}-\d{2}-\d{3}\.csv\Z"), "append_only_record"),
    (
        re.compile(r"\.agents/state-events/\d{4}-\d{2}/STATE-[A-Za-z0-9-]+\.md\Z"),
        "append_only_record",
    ),
)

CHECKER_EVIDENCE_OVERRIDES = {
    "scripts/check-agent-record.py": "tests/scripts/test_agent_record.py",
    "scripts/check-doc-checkpoint.py": "tests/scripts/test_doc_checkpoint.py",
    # Its suite predates the test_check_<name> convention and CI runs it under
    # the older name, so the derived path pointed at a file that does not
    # exist and NO change to this checker could ever satisfy its own evidence
    # rule. Mapped to the file CI actually runs.
    "scripts/check-device-leakage.py": "tests/scripts/test_device_leakage.py",
    "scripts/check-release-workflow.py": "tests/scripts/test_release_pipeline.py",
}

DISABLED_CREATION_CHECKER = (
    b"#!/usr/bin/env python3\n"
    b'\"\"\"Deliberately disabled creation-contract mutation.\"\"\"\n'
)
CREATION_MUTATIONS = {
    "scripts/check-commit-trailers.py": (
        b"#!/usr/bin/env python3\n"
        b'"""Deliberately disabled creation-contract mutation."""\n'
        b"def parsed_trailers(message): return ''\n"
        b"def validate_commit_message(message, *, strict): return []\n"
        b"def validate_range(*args, **kwargs): return []\n"
    ),
    "scripts/check-arm-isa-build.py": DISABLED_CREATION_CHECKER,
    "scripts/check-cpu-isa-build.py": DISABLED_CREATION_CHECKER,
    "scripts/check-cuda-fat-gencode.py": DISABLED_CREATION_CHECKER,
    "scripts/check-release-workflow.py": DISABLED_CREATION_CHECKER,
    "scripts/validate-release-archive.py": DISABLED_CREATION_CHECKER,
    "scripts/check-pr-size.py": DISABLED_CREATION_CHECKER,
    "scripts/check-prompt-contract.py": DISABLED_CREATION_CHECKER,
    "scripts/check-triton-aot-multiarch.py": DISABLED_CREATION_CHECKER,
    # ENG-RELEASE-WINDOWS. The portability suite loads this checker as a module
    # and calls its validation functions, so the disabled stub leaves the test
    # cases executable but red instead of inventing a permissive BASE checker.
    "scripts/check-windows-portability.py": DISABLED_CREATION_CHECKER,
    "scripts/check-windows-release-state.py": DISABLED_CREATION_CHECKER,
    # A new checker has no BASE version to mutate, so it registers the disabled
    # form its own tests must reject. The empty stub exits 0 and prints nothing,
    # which fails every case in tests/scripts/test_check_site.py -- including
    # the clean-tree case, which asserts the "nav in bijection" line.
    "scripts/check-site.py": DISABLED_CREATION_CHECKER,
    # ENG-RELEASE-CONTAINERS. Both suites load the checker as a module and call
    # into it (check_shape/check_dockerfile, validate), so the disabled stub --
    # which defines none of them -- fails every case rather than passing a
    # reduced one.
    "scripts/check-container-matrix.py": DISABLED_CREATION_CHECKER,
    "scripts/check-container-workflow.py": DISABLED_CREATION_CHECKER,
    # GATE-SYMBOL-ANCHORS (#1143). Created here, so there is no BASE version to
    # mutate. The empty stub exits 0 and prints nothing, which fails 20 of the
    # 21 cases in tests/scripts/test_check_symbol_anchors.py -- including the
    # clean-tree case, which asserts a checked count at or above the recorded
    # floor and so cannot be satisfied by silence.
    "scripts/check-symbol-anchors.py": DISABLED_CREATION_CHECKER,
    # ROCM-HW-DP4A. Created in the same pull request, so there is no BASE
    # version to mutate. Its suite imports the checker as a module and calls
    # `check(root=...)`, which the disabled stub does not define, so all 6 cases
    # in tests/scripts/test_check_rocm_dp4a_intrinsic.py error rather than a
    # reduced subset passing. Verified against the stub: "Ran 6 tests" then
    # "FAILED (errors=6)".
    "scripts/check-rocm-dp4a-intrinsic.py": DISABLED_CREATION_CHECKER,
    # 2026-08-16: the CUDA arch-gate registration guard (#960). Created in the
    # same PR, so there is no BASE version to mutate; its own suite loads the
    # checker as a module and calls into it, so the disabled stub fails at import
    # rather than quietly passing a reduced set of cases.
    "scripts/check-cuda-op-arch-gate.py": DISABLED_CREATION_CHECKER,
    # GATE-CONFLICT-MARKERS (#1417). Created in this range, so there is no BASE
    # version to mutate. The empty stub exits 0 and prints nothing, which fails
    # 16 of the 21 cases in tests/scripts/test_check_conflict_markers.py --
    # measured, not asserted. The five survivors are the four OrdinaryTextTests
    # cases, which assert only that an ordinary document exits 0 and are
    # therefore satisfied by silence, and the registration case, which reads two
    # files and never runs the checker at all. Every case that reads an exit
    # code of 1 or an examined count out of the report goes red, which is what
    # makes the stub a mutation rather than a weaker checker.
    "scripts/check-conflict-markers.py": DISABLED_CREATION_CHECKER,
    # KERNEL-ATTN-DENSE-FLASH (#1544). Created in this range, so there is no BASE
    # version to mutate. Its suite loads the checker as a module at import time and
    # then calls scan_file / has_marker / drift_sites / stale_allowlist_entries /
    # main on it, none of which the stub defines, so every case that touches the
    # checker raises AttributeError. Measured: 31 of 31 red under the stub, because
    # the suite has no case that passes without calling into the checker at all.
    "scripts/check-attention-rung-consistency.py": DISABLED_CREATION_CHECKER,
    # MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm (#2411). Created in this
    # range, so it has no BASE version to mutate, and without this entry the
    # evidence run raised "absent at BASE and has no closed creation mutation"
    # -- an absent checker reading as a broken one, which is the same shape the
    # deletion exemption above answers from the other side. Measured against the
    # stub, not asserted: "Ran 28 tests" then "FAILED (failures=19, errors=19)",
    # with ZERO cases surviving. The suite loads the checker as a module and
    # every case either calls into it (official_names, classify, vision_shape,
    # summary, verify_offline -- none of which the stub defines) or runs a copy
    # of it over mutated fixtures and reads an exit code of 1 and a counted
    # number of disagreements, which silence cannot satisfy. The import itself
    # still succeeds on the stub, deliberately: the suite touches no checker
    # attribute at module level, so the cases run and fail individually rather
    # than reporting "Ran 0 tests", which this contract reads as no evidence.
    "scripts/check-deepseek-v4-vision-manifests.py": DISABLED_CREATION_CHECKER,
}
SELF_CHECKER = "scripts/check-pr-size.py"
EVIDENCE_TIMEOUT_SECONDS = 120
TEST_COUNT = re.compile(r"Ran ([0-9]+) tests? in ")


@dataclass(frozen=True)
class ChangedPath:
    path: str
    added: int | None
    removed: int | None

    @property
    def lines(self) -> int | None:
        if self.added is None or self.removed is None:
            return None
        return self.added + self.removed


@dataclass(frozen=True)
class EvidenceResult:
    checker: str
    test_module: str
    head_tests: int
    head_passed: bool
    base_tests: int
    base_failed: bool
    detail: str = ""


def _canonical_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        bool(path)
        and not candidate.is_absolute()
        and "\\" not in path
        and "//" not in path
        and candidate.as_posix() == path
        and all(part not in {"", ".", ".."} for part in candidate.parts)
    )


def retired_class(path: str) -> str | None:
    """The class a RETIRED surface keeps, or None when the path is not retired."""

    known = RETIRED_PATHS.get(path)
    if known is not None:
        return known
    for pattern, path_class in RETIRED_PATTERNS:
        if pattern.fullmatch(path):
            return path_class
    return None


def classify_path(path: str) -> str:
    """Return one closed path class or reject an unknown/noncanonical path."""

    if not _canonical_path(path):
        raise ValueError(f"noncanonical repository path {path!r}")
    # Ahead of every other rule: a generated artifact under src/ would otherwise
    # fall through to `product` and spend a human-review budget on hex.
    if path in GENERATED_FILES:
        return "generated"
    retired = retired_class(path)
    if retired is not None:
        return retired
    if path in POLICY_FILES:
        return "policy"
    if path in APPEND_ONLY_FILES:
        return "append_only_record"
    if path in PROJECT_RECORD_FILES:
        return "project_record"
    if ISSUE_RECORD.fullmatch(path):
        return "project_record"
    if path == ".agents/upstream-inventory.json":
        return "project_record"
    if (
        path in PROCEDURE_FILES
        or SPEC.fullmatch(path)
        or CLAIM.fullmatch(path)
        or ORACLE.fullmatch(path)
        or COMPLETED.fullmatch(path)
        or COMPLETED_STATE_EVENT.fullmatch(path)
    ):
        return "procedure"
    if (
        path == STATE_MIGRATION_MANIFEST
        or STATE_MIGRATION_MANIFEST_ARCHIVE.fullmatch(path)
        or SPEC_EVIDENCE.fullmatch(path)
        or SYNC_RECORD.fullmatch(path)
        or BENCH_EVIDENCE.fullmatch(path)
        or BENCH_EVIDENCE_RUN.fullmatch(path)
        or path in RECORDED_BENCH_EVIDENCE
        or AGENT_RUN_SCRIPT.fullmatch(path)
    ):
        return "evidence"
    if path in GOVERNANCE_SUPPORT_FILES:
        return "governance_support"
    if DESIGN.fullmatch(path):
        return "design"
    if SITE_ASSET.fullmatch(path):
        return "asset"
    if path in PUBLIC_DOCUMENT_FILES or DOC.fullmatch(path) or SITE.fullmatch(path):
        return "public_document"
    if path in PRODUCT_CHECKER_FILES:
        return "product"
    if CHECKER.fullmatch(path):
        return "governance_checker"
    if CHECKER_TEST.fullmatch(path):
        return "governance_test"
    if CI.fullmatch(path):
        return "ci"
    if HOOK.fullmatch(path):
        return "ci"
    if ASSET.fullmatch(path) or RELEASE_MANIFEST_FIXTURE.fullmatch(path):
        return "asset"
    if path.startswith("third_party/"):
        return "vendored_dependency"
    if path in {
        "release/manifest-v1.schema.json",
        "release/release-matrix.json",
        # The single immutable prerelease identity consumed by planning,
        # workflows, and the post-publication audit (#117). Exact-path only:
        # mutable or future release state must still fail closed until classified.
        "release/release-version.json",
        "release/container-matrix.json",
        "scripts/env-doc-allowlist.txt",
    }:
        return "configuration"
    if path in {
        "CMakeLists.txt", ".env.example", ".gitignore", ".dockerignore",
        ".clang-format", ".gitattributes", "flake.lock", "flake.nix",
        "LICENSE", "NOTICE",
    } or re.fullmatch(r"docker/Dockerfile(?:\.[A-Za-z0-9_.-]+)?", path) or re.fullmatch(
        r"docker/[A-Za-z0-9_.-]+\.sh", path
    ):
        # The suffixed form covered docker/Dockerfile.arm64. ENG-RELEASE-CONTAINERS
        # adds the unsuffixed multi-lane docker/Dockerfile and its healthcheck
        # script, and this function FAILS CLOSED, so leaving them out rejected the
        # whole change rather than misclassifying it.
        return "configuration"
    if path.startswith(("src/", "include/", "examples/", "tools/", "cmake/", "tests/", "scripts/", "benchmarks/", "triton_kernels/")):
        return "product"
    raise ValueError(f"unclassified repository path {path!r}")


def recognized_evidence(checker_path: str) -> str:
    override = CHECKER_EVIDENCE_OVERRIDES.get(checker_path)
    if override is not None:
        return override
    name = Path(checker_path).stem.removeprefix("check-").replace("-", "_")
    return f"tests/scripts/test_check_{name}.py"


def requires_reviewed_pr(path: str) -> bool:
    """Return whether a canonical governed path requires reviewed PR arrival."""

    return classify_path(path) in PATH_CLASSES


def parse_numstat(output: str) -> list[ChangedPath]:
    changes: list[ChangedPath] = []
    seen: set[str] = set()
    for number, line in enumerate(output.splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"numstat line {number} must have exactly three fields")
        added_raw, removed_raw, path = fields
        if path in seen:
            raise ValueError(f"duplicate numstat path {path!r}")
        seen.add(path)
        classify_path(path)
        if added_raw == removed_raw == "-":
            changes.append(ChangedPath(path, None, None))
            continue
        if not added_raw.isascii() or not added_raw.isdecimal() or not removed_raw.isascii() or not removed_raw.isdecimal():
            raise ValueError(f"numstat line {number} has invalid line counts")
        changes.append(ChangedPath(path, int(added_raw), int(removed_raw)))
    return changes


def change_errors(
    changes: list[ChangedPath],
    *,
    evidence_results: dict[str, EvidenceResult] | None = None,
    deleted: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    changed_paths = {change.path: change for change in changes}
    for change in changes:
        try:
            path_class = classify_path(change.path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        # NO BINARY GUARD. A `lines is None` path used to error here as "not
        # reviewable as text" (GATE-PR-SIZE-BINARY, #615). Retired 2026-08-13:
        # it made every golden-bearing PR unmergeable by construction -- parity
        # goldens are binary by nature -- while adding nothing, because the
        # protection that matters is classification, which runs directly above
        # and still refuses any path without a class. The classifier was always
        # built to give binaries a class; see the `SITE_ASSET` note. Note
        # `lines is None` still matters downstream: the evidence contract below
        # tests it, so a binary cannot serve as mutation evidence.
        if path_class == "governance_checker":
            # A checker that is GONE at head owes no mutation evidence. The
            # contract asks the paired suite to fail against the BASE checker and
            # pass against the HEAD one; a deletion has no head checker to run,
            # and when the suite goes in the same change there is no module to
            # import either, so the contract collapsed into a ModuleNotFoundError
            # reported as "HEAD checker/test pair failed" -- an absent checker
            # reading as a broken one. You cannot mutate a guarantee that no
            # longer exists: the deletion IS the change, and reviewing it is the
            # evidence. Keyed on the CHECKER's absence, never on the suite's, and
            # never on the diff looking deletion-shaped, so a live checker cannot
            # borrow the exemption (#2290).
            if deleted is not None and change.path in deleted:
                continue
            evidence = recognized_evidence(change.path)
            evidence_change = changed_paths.get(evidence)
            if evidence_change is None or evidence_change.lines is None or evidence_change.lines <= 0:
                errors.append(
                    f"checker change {change.path!r} requires semantic mutation evidence in {evidence}"
                )
            elif evidence_results is not None:
                proof = evidence_results.get(change.path)
                if proof is None:
                    errors.append(
                        f"checker change {change.path!r} has no executable mutation result"
                    )
                elif proof.checker != change.path:
                    errors.append(f"checker evidence identity mismatch for {change.path!r}")
                elif proof.test_module != evidence.removesuffix(".py").replace("/", "."):
                    errors.append(f"checker evidence test mismatch for {change.path!r}")
                elif proof.head_tests <= 0:
                    errors.append(f"checker change {change.path!r} executed no HEAD tests")
                elif not proof.head_passed:
                    errors.append(f"HEAD checker/test pair failed for {change.path!r}: {proof.detail}")
                elif proof.base_tests <= 0:
                    errors.append(f"checker change {change.path!r} executed no BASE mutation tests")
                elif not proof.base_failed:
                    errors.append(
                        f"BASE checker stayed green for {change.path!r}; changed test is not semantic evidence"
                    )
    return errors


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    )


def resolve_commit(repo: Path, revision: str) -> str:
    if not revision or "\x00" in revision or "\n" in revision:
        raise ValueError(f"invalid revision {revision!r}")
    try:
        oid = git(
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision}^{{commit}}",
            repo=repo,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"could not resolve revision {revision!r}") from exc
    if re.fullmatch(r"[0-9a-f]{40}", oid) is None:
        raise ValueError(f"revision {revision!r} did not resolve to one commit")
    return oid


def range_base(repo: Path, base_oid: str, head_oid: str) -> str:
    """The merge base of base and head -- i.e. what a pull request actually is.

    This used to demand that `base_oid` be an ANCESTOR of head (#773). CI passes
    `pull_request.base.sha`, the TIP of the base branch, which stops being an
    ancestor the moment main advances after the branch was cut -- continuously,
    on this repo. So the checker aborted BEFORE classifying anything, and no
    fork PR was ever checked. The sibling trailer gate aborted the same way,
    which is why CI has never validated an external contributor's trailers.

    Diffing two-dot against a moved main is not merely stricter, it is WRONG:
    main's own commits appear as reversions inside the contributor's diff, so
    paths they never touched get classified and charged to them. `git diff
    A...B` is defined as `git diff $(git merge-base A B) B` and is what GitHub
    itself shows.

    Unrelated histories still fail closed. No merge base means there is no range
    to compute, and reporting one would be an invention -- absence of
    information must never look like absence of work.
    """
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", base_oid, head_oid],
        capture_output=True,
        text=True,
        timeout=EVIDENCE_TIMEOUT_SECONDS,
        shell=False,
    )
    if result.returncode != 0:
        raise ValueError("base and head have no merge base (unrelated histories)")
    oid = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", oid) is None:
        raise ValueError("merge base did not resolve to one commit")
    return oid
    if result.returncode != 0:
        raise ValueError("could not establish base/head ancestry")


def changed_paths(base: str, head: str, *, repo: Path = ROOT) -> list[ChangedPath]:
    base_oid = resolve_commit(repo, base)
    head_oid = resolve_commit(repo, head)
    # From the MERGE BASE, not the base tip -- see range_base() (#773).
    return parse_numstat(
        git("diff", "--no-renames", "--numstat",
            range_base(repo, base_oid, head_oid), head_oid, repo=repo)
    )


def deleted_paths(base: str, head: str, *, repo: Path = ROOT) -> set[str]:
    """Paths the range DELETES, from git's own rename-aware diff filter.

    Read from `--diff-filter=D` rather than inferred from a numstat row that
    happens to add nothing: a file emptied but kept is not deleted, and a rename
    is not a deletion either. Asking git keeps the two apart.
    """

    base_oid = resolve_commit(repo, base)
    head_oid = resolve_commit(repo, head)
    return {
        line
        for line in git(
            "diff", "--no-renames", "--diff-filter=D", "--name-only",
            range_base(repo, base_oid, head_oid), head_oid, repo=repo,
        ).splitlines()
        if line
    }


EVIDENCE_REQUIRED_TOOLS = {
    # pwsh: `check-windows-portability.py` executes `pwsh` for its PowerShell
    # AST stage (scripts/check-windows-portability.py:2028) and skips the stage
    # when `shutil.which("pwsh")` misses. The evidence lane's sanitized PATH is
    # `os.defpath` (`/bin:/usr/bin`) plus the private tools directory
    # (`_sanitized_env` below), so the pr-size job's tarball install at
    # `/usr/local/bin/pwsh` (.github/workflows/ci.yml) is INVISIBLE to both
    # lanes until it is declared here: a changed test whose red runs through
    # the AST stage then stays green at BASE and the gate rejects the evidence
    # with "BASE checker stayed green". Declaring it puts the same pwsh in
    # HEAD's and BASE's PATH, which is the toolchain symmetry ci.yml installs
    # pwsh for.
    "tests.scripts.test_check_windows_portability": ("cmake", "ninja", "pwsh"),
    # #1892: this module CONFIGURES CMake in nearly every case and uses the
    # `Ninja Multi-Config` generator in one of them. Without the declaration it
    # ran with an empty private tools directory and died with
    # `FileNotFoundError: 'cmake'` -- 26 errors reported against whichever
    # checker was under change, which is a broken instrument returning a code
    # verdict. Same gap #458 closed for the module above.
    "tests.scripts.test_check_test_registration": ("cmake", "ctest", "ninja"),
}



def _prepare_evidence_tools(container: Path, module: str) -> Path:
    """Expose exact host tools privately without inheriting their PATH."""

    tools = container / "tools"
    tools.mkdir()
    for name in EVIDENCE_REQUIRED_TOOLS.get(module, ()):
        source = shutil.which(name)
        if source is None:
            raise ValueError(f"semantic evidence requires executable {name}")
        resolved = Path(source).resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ValueError(f"semantic evidence tool is not executable: {resolved}")
        (tools / name).symlink_to(resolved)
    return tools


def _sanitized_env(home: Path, tools: Path | None = None) -> dict[str, str]:
    path = os.defpath
    if tools is not None:
        path = str(tools) + os.pathsep + path
    return {
        "PATH": path,
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def _run_test_module(
    worktree: Path, module: str, tools: Path | None = None
) -> tuple[int, bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", module],
            cwd=worktree,
            env=_sanitized_env(worktree, tools),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=EVIDENCE_TIMEOUT_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"semantic evidence timed out for {module}") from exc
    counts = [int(value) for value in TEST_COUNT.findall(result.stdout)]
    if len(counts) != 1 or counts[0] <= 0:
        raise ValueError(f"semantic evidence did not execute tests for {module}")
    return counts[0], result.returncode == 0, result.stdout[-2000:]


def _base_checker(repo: Path, base_oid: str, checker_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{base_oid}:{checker_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=EVIDENCE_TIMEOUT_SECONDS,
        shell=False,
    )
    if result.returncode == 0:
        return result.stdout
    if checker_path not in CREATION_MUTATIONS:
        raise ValueError(
            f"{checker_path} is absent at BASE and has no closed creation mutation"
        )
    return CREATION_MUTATIONS[checker_path]


def executable_evidence(
    repo: Path,
    base: str,
    head: str,
    changes: list[ChangedPath],
) -> dict[str, EvidenceResult]:
    """Prove each checker change red-before/green-after in an isolated worktree."""

    base_oid = resolve_commit(repo, base)
    head_oid = resolve_commit(repo, head)
    # The BASE version of a checker, for the red-before half, is the one at the
    # merge base -- not at a base tip that has moved past this branch (#773).
    base_oid = range_base(repo, base_oid, head_oid)
    changed = {item.path for item in changes}
    checkers = sorted(
        item.path
        for item in changes
        if classify_path(item.path) == "governance_checker"
    )
    results: dict[str, EvidenceResult] = {}
    for checker_path in checkers:
        evidence_path = recognized_evidence(checker_path)
        if evidence_path not in changed:
            continue
        module = evidence_path.removesuffix(".py").replace("/", ".")
        container = Path(
            tempfile.mkdtemp(prefix="vllm-policy-evidence-", dir="/dev/shm")
        )
        worktree = container / "worktree"
        try:
            tools = _prepare_evidence_tools(container, module)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "worktree",
                    "add",
                    "--quiet",
                    "--detach",
                    str(worktree),
                    head_oid,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=EVIDENCE_TIMEOUT_SECONDS,
                shell=False,
                env=_sanitized_env(container),
            )
            head_count, head_passed, head_detail = _run_test_module(
                worktree, module, tools
            )
            target = worktree / checker_path
            target.write_bytes(_base_checker(repo, base_oid, checker_path))
            target.chmod(0o755)
            base_count, base_passed, base_detail = _run_test_module(
                worktree, module, tools
            )
            results[checker_path] = EvidenceResult(
                checker=checker_path,
                test_module=module,
                head_tests=head_count,
                head_passed=head_passed,
                base_tests=base_count,
                base_failed=not base_passed,
                detail=head_detail if not head_passed else base_detail,
            )
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=EVIDENCE_TIMEOUT_SECONDS,
                shell=False,
                env=_sanitized_env(container),
            )
            shutil.rmtree(container, ignore_errors=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--branch", default="", help="accepted for CI compatibility")
    parser.add_argument("--pr-number", default="")
    args = parser.parse_args()
    del args.branch
    try:
        if args.pr_number and (
            not args.pr_number.isascii()
            or not args.pr_number.isdecimal()
            or int(args.pr_number) <= 0
        ):
            raise ValueError("--pr-number must be a positive decimal integer")
        base_oid = resolve_commit(ROOT, args.base)
        head_oid = resolve_commit(ROOT, args.head)
        changes = changed_paths(base_oid, head_oid)
        gone = deleted_paths(base_oid, head_oid)
        evidence = executable_evidence(
            ROOT, base_oid, head_oid,
            [c for c in changes if c.path not in gone],
        )
        errors = change_errors(
            changes,
            evidence_results=evidence,
            deleted=gone,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: PR size check could not classify the change: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: every explicit path class is within its review budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
