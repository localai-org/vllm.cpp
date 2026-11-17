#!/usr/bin/env python3
"""Classify each gated row's Gates section: does it name a command that can FAIL?

A row's `Gates` field promises "exact commands", and nothing has ever checked
that one exists or that it can fail. A gate that is `true`, `echo ok`, or piped
into another command collapses "done" into the implementer's opinion of its own
work.

This ships as a CLASSIFIER first and a ratchet second, deliberately: most gated
rows cannot state a runnable command today, so a gate demanding one would be red
on arrival and would have to be relaxed to pass. A relaxed gate is worse than no
gate.

The ratchet is the second half, wired only AFTER the debt was recorded
(.agents/specs/gate-command-audit-2026-08-06.md), so it ships green and never had
to be relaxed to pass. It pins the SET of rows carrying a runnable command, and
it is an EXACT PIN, not a shrink-only floor: `--check` below refuses a row that
LOST its command, and tests/scripts/test_check_gate_commands.py additionally
asserts RUNNABLE_BASELINE equals the shipped set, so growth is red too. That is
deliberate -- an exact pin is what makes "just lower the number" impossible --
and it means ANY movement, up or down, re-pins RUNNABLE_BASELINE in the SAME
change, naming the rows and the reason. Growth is welcome; silent growth is not.

    scripts/check-gate-commands.py            # report
    scripts/check-gate-commands.py --json     # machine-readable
    scripts/check-gate-commands.py --check    # gate: no row may lose its command
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


record = _load("agent_record", "scripts/check-agent-record.py")

# The runnable-command audit follows work that can still move. DONE rows have
# immutable closing evidence and leave this population; keeping them here would
# turn a closure into a permanent baseline entry instead of auditing live debt.
GATED_STATES = frozenset({"READY", "ACTIVE", "GATING", "BLOCKED"})

# check-agent-record.py's MATRIX_PATHS covers 5 of the 7 matrices. feature-matrix
# is added here without widening that constant -- it governs a repo-wide CI gate
# whose row contract these two files have never been held to.
#
# sglang-matrix.md is DELIBERATELY ABSENT, and the reason is recorded rather than
# implied (.agents/specs/gate-command-audit-2026-08-06.md risk 6). Step 2 listed
# it; it contributed 0 rows of its 87 table rows, with 0 parse errors, because it
# carries a CLASSIFICATION column (FUSED / INVENTORIED / NOT-APPLICABLE) in place
# of a lifecycle state, so parse_claim_rows recognises nothing in it. Listed and
# empty is this repo's recorded defect class -- a reader of this constant would
# conclude SGLang rows were examined and found clean. It has no gated rows to
# audit, so it is not audited; the test pins that justification, and goes red if
# the matrix ever gains lifecycle rows.
AUDITED_MATRIX_PATHS = [
    *record.MATRIX_PATHS,
    record.AGENTS / "feature-matrix.md",
]

_GATES_HEADING = re.compile(r"(?im)^#{1,6}\s*gates\b.*$")
_HEADING = re.compile(r"(?m)^#{1,6}\s")

# A command names an executable: a known tool as a WHOLE WORD, or a path that is
# actually INVOKED. A backticked filename is not a command -- `docs/BENCHMARKS.md`
# and `tests/vllm/models/test_model_registry.cpp` are things a gate talks about,
# not things it runs. Both boundaries matter on the shipped record: without the
# trailing one `sha256_cbor` matches `sh` and `python@3.14` matches `python`.
_TOOL = re.compile(
    r"(?:^|\s)(?:ctest|pytest|python3?|cmake|bash|sh|make|nsys|ncu|git|gh)(?:\s|$)"
)
# `flock <lock> -c '<gate>'` is this repo's MANDATED shape for any gate touching
# the GPU, and the wrapper QUOTES the real command, putting it out of reach of
# every other rule here. It needs a lockfile AND something to run: a bare
# `flock`, and a `flock` plus a lockfile with nothing after it, appear in three
# specs naming the idiom rather than a gate, and a plain vocabulary entry would
# credit all three with a command.
_WRAPPER = re.compile(r"(?:^|\s)flock\s+\S+\s+\S")
# `./anything` is an explicit invocation, arguments or not -- a built test binary
# (`./build-cuda-121a/tests/test_dropin_abi`) is run, not referred to. A bare
# `scripts/`/`tests/` path is only a command when it carries an executable suffix
# or arguments; otherwise it is a filename.
_INVOKED_PATH = re.compile(
    r"(?:^|\s)(?:\./\S+|(?:scripts|tests)/\S*(?:\.(?:py|sh)(?:\s|$)|\s+\S))"
)
# Shapes that cannot fail, so they are not gates at all.
_CANNOT_FAIL = re.compile(r"^\s*(true|:|echo\b)")


def gates_section(text: str) -> str | None:
    """The body under the first `Gates` HEADING, or None. Prose does not count."""
    match = _GATES_HEADING.search(text)
    if not match:
        return None
    rest = text[match.end() :]
    nxt = _HEADING.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _candidates(section: str) -> list[str]:
    inline = re.findall(r"`([^`\n]+)`", section)
    fenced = re.findall(r"```[a-z]*\n(.*?)```", section, re.S)
    for block in fenced:
        inline.extend(line for line in block.splitlines() if line.strip())
    return [c.strip() for c in inline if c.strip()]


def is_command(candidate: str) -> bool:
    """Does this backticked span name something you could RUN at all?

    The no-op shells (`true`, `:`, `echo ...`) are commands, and are recognised
    here DELIBERATELY: `runnable_commands` must reject them for the reason that
    matters -- they cannot fail -- and not merely fail to notice them. A
    classifier that never sees `true` pins nothing about the rule it exists for.
    """
    padded = " " + candidate
    return bool(
        _TOOL.search(padded)
        or _WRAPPER.search(padded)
        or _INVOKED_PATH.search(padded)
        or _CANNOT_FAIL.match(candidate)
    )


def runnable_commands(section: str) -> list[str]:
    """Commands in this section that could actually fail."""
    good = []
    for candidate in _candidates(section):
        if not is_command(candidate):
            continue
        if _CANNOT_FAIL.match(candidate):
            continue
        if "|" in candidate:  # `cmd | tail` reports tail's status
            continue
        good.append(candidate)
    return good


# A `Gates` section is a numbered list, and each item opens with a BOLD LEAD that
# titles it. These three functions exist so a test can key on that structure
# instead of on a sentence -- see .agents/specs/fix-gate-commands-prose-pin.md.
#
# They are DELIBERATELY not wired into `ratchet_errors`. A rule saying every gate
# item without a runnable command must declare a disposition is red on arrival for
# items this repo writes in prose rather than quoting (`Red first.`,
# `CUDA compile.`), and this file's own header records that a gate which has to be
# relaxed to pass is worse than no gate. That sweep is owed with its own survey.
_GATE_ITEM = re.compile(r"(?m)^\d+\.\s")
# The lead is a single-line title. A lead that does not close on its own line
# returns None, which reads downstream as "declares no disposition" and takes a
# gate RED rather than green. That is the safe direction for an unparsed record.
_ITEM_LEAD = re.compile(r"^\d+\.\s+\*\*(.+?)\*\*")
# A CLOSED, small vocabulary, matched in the LEAD ONLY. A status word in the lead
# is a declaration ABOUT the gate. The same word in the body is ordinary prose
# describing what the gate does, and crediting it would make this stop detecting
# silence. Scope is MEASURED, not assumed: surveyed over every `Gates` section in
# .agents/specs/ on 2026-08-18, 32 of 323 gate items GAIN a disposition when the
# search widens from the lead to the whole item -- `**No regression:**` in
# cpu-elementwise-gemm.md and `**Correctness gate:**` in dropin-kernel-abi.md both
# pick up `pass` out of body prose. The row that motivated this file is NOT one of
# the 32; that was the first hypothesis and the survey refuted it.
_DISPOSITION = re.compile(
    r"(?i)\b(owed|waived|blocked|deferred|superseded|not gated|ran|pass|passed|failed)\b"
)


def gate_items(section: str) -> list[str]:
    """The numbered items of a `Gates` section, in order."""
    starts = [m.start() for m in _GATE_ITEM.finditer(section)]
    ends = starts[1:] + [len(section)]
    return [section[a:b].rstrip() for a, b in zip(starts, ends)]


def item_lead(item: str) -> str | None:
    """The bold lead that titles one gate item, or None if it has no closed lead."""
    match = _ITEM_LEAD.match(item)
    return match.group(1) if match else None


def gate_disposition(item: str) -> str | None:
    """The status this gate item DECLARES in its lead, or None if it declares none.

    `None` is the finding this exists to report: a gate item that names no
    disposition and yields no runnable command is a leg the record is silent
    about, and a reader of a credited row infers coverage nobody claimed.
    """
    lead = item_lead(item)
    if lead is None:
        return None
    match = _DISPOSITION.search(lead)
    return match.group(1) if match else None


def classify_row(row) -> tuple[str, str]:
    specs = [p for p in record.local_spec_paths(row) if p.is_file()]
    if not specs:
        return "no-spec", "no resolving .agents/specs/ link"
    text = specs[0].read_text(encoding="utf-8", errors="replace")
    section = gates_section(text)
    if section is None:
        return "no-gates-section", specs[0].name
    commands = runnable_commands(section)
    if not commands:
        return "gates-no-command", specs[0].name
    return "runnable", commands[0]


class RecordParseError(RuntimeError):
    """A matrix did not parse, so the audit below it is INCOMPLETE.

    This existed as a silent `errors` list nobody read, and the consequence was
    exactly this repo's recorded defect class: strip rows from a matrix and the
    ratchet reported `these baseline rows left the gated population ... re-pin
    RUNNABLE_BASELINE` -- a parse FAILURE wearing the face of a legitimate
    record edit, recommending the one action the audit says must never be taken
    blindly. A parse failure and a record edit must never look the same.
    """

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = list(errors)


def audit() -> list[dict]:
    """Classify every gated row. Raises RecordParseError if a matrix is broken."""
    records = []
    errors: list[str] = []
    for path in AUDITED_MATRIX_PATHS:
        for row in record.parse_claim_rows(path, errors):
            if row.state not in GATED_STATES:
                continue
            verdict, detail = classify_row(row)
            records.append(
                {
                    "id": row.item_id,
                    "state": row.state,
                    "path": str(row.path.relative_to(ROOT)),
                    "line": row.line_no,
                    "verdict": verdict,
                    "detail": detail,
                }
            )
    if errors:
        raise RecordParseError(errors)
    return records


# An EXACT PIN over a SET of row IDs -- not a count, and NOT shrink-only.
#
# Not a count, because a count cannot tell "this row lost its gate command" from
# "this row left the population", and the population moves: 3 rows moved
# mid-branch while the classifier above was being written, which is why the total
# (97) was deliberately never pinned. Pinning a count would go red on a legitimate
# record edit, and the natural "fix" is to lower the number, which is the gate
# erasing its own finding.
#
# Not shrink-only, because the pin is enforced from BOTH sides: `ratchet_errors`
# below catches a row that lost its command, and
# tests/scripts/test_check_gate_commands.py asserts this frozenset EQUALS the
# shipped runnable set, which is what makes lowering the baseline impossible to
# do quietly. The same equality means GROWTH is red too: add a real gate command
# to a row's spec and `--check` stays 0 while the suite, preflight and CI go red
# until this set is re-pinned. That is the intended cost. Growth is ordinary,
# welcome work -- transcribe a row's existing evidence into an invocation -- but
# ANY movement, up or down, re-pins RUNNABLE_BASELINE in the SAME change, naming
# the rows that moved and why.
#
# This is a pin, not a certificate: five of these credits are weak (two MLX
# `pip install` lines, `git diff --check`, TOOLS-STREAMING-PARSER resting solely
# on `git diff --stat`, which exits 0 unconditionally in a repo, and
# KERNEL-GEMM-CPU-ELEM credited a bare `ctest -j2` lifted from prose describing a
# FLAKE). They are pinned anyway -- see
# .agents/specs/gate-command-audit-2026-08-06.md risk 3. A ratchet that waits for
# a clean baseline never starts.
# 2026-08-10: +ENG-RELEASE-CONTAINERS enters the runnable population when its
# spike spec lands (issue #170). The credit is INHERITED, not container-specific,
# and the distinction matters: the row's own gates -- the image layout audit, the
# container smoke, `scripts/check-container-workflow.py` -- do not exist yet, and
# nothing about a container is executed by anything in the tree today. What the
# spec does bind is that every image build runs the release chain it already
# depends on (`scripts/build-*-release.sh`, which end in
# `scripts/validate-release-archive.py`), plus the staged-tree contract in
# `tests/scripts/test_server_package.py`. Those exist and genuinely fail on a
# broken staged tree, so the row is credited rather than pinned as
# gates-no-command -- but this is the weakest kind of credit in the set and it
# is re-earned, not re-confirmed, when W3 lands the container-specific gates.
# 2026-08-09: +ENG-DOCS-SITE enters the runnable population. Its spec's Gates
# section carries `python3 scripts/check-site.py` and `hugo --minify`, both of
# which genuinely fail on a broken site, so it is credited on arrival rather
# than pinned as gates-no-command. Issue #224.

# 2026-08-09: +SAMPLE-PROMPT-LOGPROBS. The row reached ACTIVE with the runner
# prompt-logits source (#223) and its spec's Gates section carries the exact
# configure/build/focused-test/full-ctest invocation the gate was run with,
# including the serial re-run for the known parallel-ctest flake. Growth, so the
# set is re-pinned in the same change.
# 2026-08-10: +LORA-RUNTIME enters the runnable population. It did not gain a
# gate; it re-entered the AUDITED population when the row went back to `ACTIVE`
# for W2 (issue #278) after the 2026-08-04 triage parked it at ANCHOR-BACKFILL.
# Its only pre-existing credit was the UPSTREAM path
# `tests/lora/test_qwen35_densemodel_lora.py` named in prose as the eventual
# model gate -- one of the weak credits described above. The same change adds
# the row's REAL invocation (the CPU configure/build plus
# `ctest -R test_punica_cpu` and `ctest -R test_lora_layers`), so the pin rests
# on a command that genuinely fails when the row regresses.
#
# 2026-08-11: +ENG-RECORD-CONFLICT-SURFACES. The row reaches READY on its
# committed spec (issue #364), whose Gates section names the exact preflight,
# `tests/scripts/` and `agent-integration.py` invocations the record gate runs
# with, and records that no CUDA/GPU/SACRED gate is implicated because no product
# source is touched. Growth, so the set is re-pinned in the same change.
# 2026-08-11: -ENG-ASYNC-SCHED, -SERVE-HTTP-TRANSPORT and
# -ENG-NOW-DERIVED. DONE is closed evidence, not live gated work; #374 exposed
# that retaining DONE made a completed protocol row a permanent runnable
# baseline member. All three departures are the same lifecycle-policy closure,
# not downgraded verdicts or hidden work. Re-adding DONE to GATED_STATES is the
# load-bearing mutation pinned in the paired suite.
# 2026-08-11: +ENG-TRAILER-MERGE-ARTIFACTS on arrival at ACTIVE, then REMOVED
# the same day on reaching DONE (closing commit 157080c8) -- a DONE row leaves
# the gated population, so its verdict is None rather than a downgraded one.
# A shrink for a real record edit, named as the message demands.
# 2026-08-21: -SPEC-BPE-QUADRATIC-MERGE. It entered on 2026-08-19 as GROWTH,
# when its `## Gates` section gained the `g++` build and the two run lines for
# `tools/bench/bpe_encode_cost.cpp`, and it leaves now because the row was
# promoted `GATING` -> `DONE` by its closing commit (fix `67823aee2`, PR #1539,
# issue #1365). Same shape and same reason as ENG-TRAILER-MERGE-ARTIFACTS above:
# `DONE` is not in GATED_STATES, so the row leaves the AUDITED population and its
# verdict becomes None rather than a downgraded one. It did NOT lose its command
# -- the recipe is still in the spec and still runnable by hand -- which is
# exactly the distinction `ratchet_errors` draws, and why this is a shrink for a
# real record edit rather than a gate erasing its own finding. A shrink, named as
# the message demands, and re-pinned in the SAME change as the promotion.
# 2026-08-11: +ENG-FORGE-COAUTHOR. Reaches ACTIVE on its committed spec (issue
# #418), whose Gates section names the preflight, tests/scripts and
# agent-integration invocations plus the per-commit re-verification of
# f64f2b71, and records that no CUDA/GPU/SACRED gate is implicated because no
# product source is touched. Growth, so the set is re-pinned in the same change.
# 2026-08-13: +SERVE-RECIPE-ARGS. The row leaves SPIKE for ACTIVE on its
# implementation (issue #606), which puts it in GATED_STATES for the first time;
# its spec's Gates section names `scripts/agent-preflight.sh --staged` plus the
# focused test file, and records that no CUDA/GPU/SACRED/oracle gate is
# implicated because the change is argument parsing and reaches no forward pass.
# Growth from a lifecycle move, so the set is re-pinned in the same change.
# 2026-08-14: +ENG-RECORD-ANCHOR-RATCHET. The row leaves SPIKE for ACTIVE on its
# implementation (issue #632), which puts it in GATED_STATES for the first time.
# Its spec's Gates section names five invocations, and this is a STRONG credit
# rather than one of the weak ones described above: the row's gate IS
# `scripts/check-agent-record.py`, so the credited command is the thing under
# test, and it fails on either direction of the ratchet. The suite
# (`tests/scripts/test_agent_record.py`) is proven red against the BASE checker
# by `scripts/check-pr-size.py`, which is itself one of the five. Growth from a
# lifecycle move, so the set is re-pinned in the same change.
# 2026-08-16: +SPEC-MTP-K-GT-1. A NEW row arriving at ACTIVE (issue #81), so it
# enters GATED_STATES for the first time. Its spec's Gates section names
# `scripts/agent-preflight.sh` plus the built CPU suite (493 passed / 0 failed /
# 2 skipped of 495, the two skips checkpoint-gated and unrelated) and the three
# focused doctest binaries, and records what is NOT implicated and why: the
# change reaches a GPU forward only through paths the CPU tier already runs, so
# no CUDA or SACRED gate is claimed here, and the DGX three-way at k=2..4 is
# recorded as OWED rather than skipped. Growth, so the set is re-pinned in the
# same change.
# 2026-08-17: +ENG-RESIDENCY-CONFIG. A NEW row arriving at ACTIVE (issue #1110),
# so it enters GATED_STATES for the first time. Its spec's Gates section names the
# documented CPU configure/build recipe, `ctest -j 6`, the three focused doctest
# binaries by name, `scripts/agent-preflight.sh --staged`, and the reachability
# mutation (delete the install call site in `LoadedEngine::FromModelDir`, require
# the server-level suite red). It also records what is NOT implicated and why: the
# row moves where a value comes from and changes no kernel, dtype, allocation or
# token, so it claims no CUDA, SACRED or throughput gate, and the 370 GiB
# reproduction through the JSON form is recorded as OWED rather than skipped
# because dgx.casa was unreachable at the SSH layer. Growth, so the set is
# re-pinned in the same change.
# 2026-08-18: +ENG-CUDAGRAPH-DEDUP. A NEW row arriving at ACTIVE (issue #1162),
# so it enters GATED_STATES for the first time. Its spec's Gates section names
# `ctest -R test_graph_dedup` and `scripts/agent-preflight.sh`, both of which
# genuinely fail when the row regresses -- the focused suite detected 9 of 9
# negative mutations of the registry it gates. It also records what is NOT
# claimed and why: the device byte-identity A/B needs a leased CUDA box this
# session did not have, so it is carried under the spec's `## Owed` rather than
# reported as run, and the CUDA leg's compile rests on the `cuda-fat-build` CI
# job. Growth, so the set is re-pinned in the same change.
# 2026-08-18: +ENG-HF-MODEL-DOWNLOAD. A NEW row arriving at READY (issue #1280),
# so it enters GATED_STATES for the first time. Its spec's Gates section names
# `scripts/validate-container-image.py`, which boots the image and asserts the
# failure for an unknown repository is an HTTP 404 from the hub rather than the
# message that names the build options. That distinction is the point: a symbol
# check passes on a build where `VLLM_CPP_HF_DOWNLOAD` resolved OFF, so the
# container gate is the only instrument that separates a working TLS build from
# a silently disabled one. The section also records what the runnable gates do
# NOT cover: the hermetic suite speaks plain hypertext transfer protocol, so it
# proves nothing about transport layer security, and the second instrument is an
# opt-in online test that does not run in the default lane. Growth, so the set
# is re-pinned in the same change.
# 2026-08-19: +ENG-CUDAGRAPH-BREAK, and this entry is a REPAIR rather than a
# landing. W5 of that row (#1361, commit 601b576c6) filled its spec's Gates
# section with runnable evidence, including a named test binary with its case
# and assertion counts and an exit status, which is what moves a row into the
# runnable population. The re-pin this ratchet's own error text demands was not
# made in that change, so `main` itself has been red on
# tests/scripts/test_check_gate_commands.py since it landed: 8 failures of 44,
# every one a comparison between the computed runnable set and this pin,
# measured in a detached worktree of origin/main. The continuous integration
# lane that would have caught it independently has not executed for this
# repository since roughly 07:43Z that day, so the change landed with no remote
# verdict. Found while merging origin/main into row/ENG-HF-MODEL-DOWNLOAD and
# fixed in that flow under #1376, because the fix is small and clear and a red
# main blocks every other row's gate. Growth, so the set is re-pinned.
# 2026-08-22: +SERVE-REQUEST-LENGTH-GUARD enters the runnable population with the
# row itself (#1541). GROWTH, so the set is re-pinned in the same change, per the
# note above. Its spec `## Gates` names a configure, a ninja target list and three
# suite invocations that genuinely fail on a broken guard -- the same three the
# implementing branch's red-then-green evidence was taken from -- so the credit is
# the row's own and not inherited.
# 2026-08-24: +ENG-UPSTREAM-LTX2-PIN enters the runnable population when its spec
# lands (#1433). GROWTH, so the set is re-pinned in the same change, per the note
# above. The credit is the row's OWN and is not inherited: its `## Gates` names
# `scripts/check-oracle-pins.py`, its `--self-test`, `scripts/check-agent-record.py`
# and two pytest suites, and the first genuinely fails on a broken record rather
# than exiting 0 in any tree. That was proved by mutation in the same change, both
# directions: deleting `.agents/oracles/ltx-2.md` reds with `admitted by the
# AGENTS.md table but has no .agents/oracles/ record`, and deleting the AGENTS.md
# table row reds with `pinned in .agents/oracles/ but absent from the AGENTS.md
# table`. A record row is credited for the checker that reads the record, which is
# the only thing a record row can execute.
# 2026-08-25: +ENG-POOL-BEST-FIT enters the runnable population when its spec
# lands (#1922). GROWTH, so the set is re-pinned in the same change, per the note
# above. The credit is the row's OWN: its `## Gates` names a configure, the
# focused target, TWO invocations of the SAME binary whose verdicts must
# DIFFER -- `./build/tests/test_engine_scratch_steady_state` must pass and
# `VT_POOL_BORROW=0 ./build/tests/test_engine_scratch_steady_state` must fail --
# then the full build and `ctest`. A gate that names both arms of a same-binary
# A/B cannot be satisfied by a tree in which the guard does nothing, which is
# exactly what "genuinely fails on a broken guard" asks for, and it was proved by
# three mutations in the same change rather than asserted.
# 2026-08-29: -MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm leaves the
# runnable population because the ROW leaves the gated population, not because it
# lost a command ([#2214](https://github.com/mudler/vllm.cpp/issues/2214)). It moves
# `BLOCKED` -> `SPIKE` on a committed port plan, and `SPIKE` is not a gate-obliged
# state, so `check-agent-record.py` stops reading the row's spec for a `## Gates`
# section and this audit stops seeing the row at all. SHRINKAGE, so the set is
# re-pinned in the same change, per the note above. Nothing was weakened: the
# `### Gates` section that earned the credit is untouched in
# .agents/specs/glm-dsa-latest-deepseek.md, and §3.6 of that spec ADDS four named
# gates (module parity against the pin on CPU, a headers-only structural loader
# gate, a streamed-vs-resident identical-logits gate, and an llama.cpp b10451
# floor) together with the statement that no end-to-end token gate against vLLM is
# reachable on this fleet. The credit returns when the row reaches a gate-obliged
# state, which its W2 does.
RUNNABLE_BASELINE = frozenset({
    # 2026-10-06: +BACKEND-XPU-EXL3-SERVING. The new ACTIVE row's committed
    # serving spec declares runnable CPU/XPU/server gates. Pin this population
    # growth in the same integration change; no existing row or gate is removed.
    "BACKEND-XPU-EXL3-SERVING",
    # 2026-09-12: +QUANT-GGUF-IQ4_NL. GROWTH, and re-pinned in the change that
    # caused it. The row is new (#3149) and its spec's `## Gates` section names
    # a `ctest` invocation and the `strix:gpu0` ROCm suite, both of which can
    # genuinely fail -- the section records this row's own RED run at
    # `3c529c87e` (2 of 47 cases) before the arm existed. A row enters this
    # population by what its Gates section can RUN, and these run.
    "QUANT-GGUF-IQ4_NL",
    # 2026-09-05: +BACKEND-TENSTORRENT-KEEPQUANT. GROWTH, and re-pinned in the
    # same change that caused it. The row is new (#2959, W0 staging) and its
    # spec's `## Gates` section names `scripts/agent-preflight.sh` per wave,
    # which can genuinely fail -- the first full preflight on this very tree
    # reded through this checker until the entry was pinned. The wave-local
    # command surfaces (op-level suite, capture identity, e2e battery) enter
    # the section as the waves land, not in the staging change.
    "BACKEND-TENSTORRENT-KEEPQUANT",
    # 2026-09-02: +QUANT-EXL3-PERF. GROWTH, and re-pinned in the same change that
    # caused it. The row is new (#2570) and its spec's `## Gates` section names
    # two `ctest` invocations and `scripts/agent-preflight.sh --staged`, each of
    # which can fail: `test_exl3_gemv` carries the envelope cases, which are pure
    # integer arithmetic and run anywhere, and the device tier-3c case, which is
    # RED for codebook 2 on any tree where that arm is not instantiated. A row
    # enters this population by what its Gates section can RUN, and these run.
    "QUANT-EXL3-PERF",
    # 2026-09-01: +QUANT-EXL3-MUL1. GROWTH, and re-pinned in the same change
    # that caused it, as this ratchet requires. The row is new (#2495) and its
    # spec's `## Gates` section names four `ctest` invocations and
    # `scripts/agent-preflight.sh --staged`, every one of which can fail: the
    # host codebook-2 decode is gated against hand-computed upstream literals,
    # and the loader and seam cases were RED on this tree before the row's
    # implementation existed. A row enters this population by what its Gates
    # section can RUN, and these run.
    "QUANT-EXL3-MUL1",
    # 2026-08-31: -ENG-PREFLIGHT-COMPILES leaves the runnable population when
    # b39f14adb moves the row ACTIVE -> PARTIAL (#2401). PARTIAL is outside this
    # lifecycle-scoped audit, so the baseline excludes the row. Its spec keeps
    # both commands that can fail; this is a lifecycle shrink, not a weaker
    # verdict or a lost gate command.
    # 2026-08-30: +MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm. GROWTH, and
    # not because the row gained a gate: it re-entered the AUDITED population
    # when W2 (#2214) moved it `SPIKE` -> `ACTIVE`, and its spec's `### Gates`
    # section already named commands that can fail. Re-pinned in the same change
    # that moved the row, as the checker requires.
    "MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm",
    # 2026-10-03: -MODEL-DSV4-DSA-COMPOSE leaves the runnable population.
    # The row is no longer a gated matrix row anywhere: its record moved to
    # .agents/completed/issue-index.md after the DeepSeek-V4 DSA composition
    # work landed through the MODEL-MM-deepseek-v4 vision row (W3 wired the
    # cr==4 arm end to end, d4d7e9573), superseding the scoping-only row the
    # entry was pinned for (#2286). Removal re-pins the baseline in its own
    # change, naming the row and the reason, as the ratchet requires.
    "ENG-POOL-BEST-FIT",
    "ENG-UPSTREAM-LTX2-PIN",
    "SERVE-REQUEST-LENGTH-GUARD",
    "ENG-CUDAGRAPH-BREAK",
    "ENG-HF-MODEL-DOWNLOAD",
    "ENG-RESIDENCY-CONFIG",
    "ENG-CUDAGRAPH-DEDUP",
    "SPEC-MTP-K-GT-1",
    "ATTN-CHUNKED-LOCAL",
    "ENG-RECORD-ANCHOR-RATCHET",
    "SERVE-RECIPE-ARGS",
    "ENG-FORGE-COAUTHOR",
    "ENG-RECORD-CONFLICT-SURFACES",
    "SAMPLE-PROMPT-LOGPROBS",
    "ATTN-ROPE-FAMILY",
    "BACKEND-CUDA-ARCH-ADDITIVITY",
    "BACKEND-METAL-MLX",
    "BACKEND-VULKAN",
    "ENG-CORE-BUSY-LOOP",
    "ENG-DOCS-SITE",
    "ENG-EXPERT-STREAM",
    "ENG-LOAD-DIRECT-UPLOAD",
    "ENG-PRIORITY-SCHED",
    "ENG-RELEASE-CONTAINERS",
    "KERNEL-GEMM-CPU-ELEM",
    # KV-EVENTS W3 (2026-08-11, issue #352): GROWTH, re-pinned in the same
    # change. The row's Gates section previously described only in-process
    # assertion counts; W3 names the CPU test binaries and the ctest sweep that
    # gate the scheduler envelope and the report_mode=="full" reuse path, so the
    # row now carries a command that can fail and joins the runnable population.
    "KV-EVENTS",
    "LORA-RUNTIME",
    "KV-CHUNKED-LOCAL-SPEC",
    "KV-SLIDING-LOCAL-SPECS",
    # ARCH-ONE-SURFACE ROW 6 (2026-08-08): SERVE remains gated, while the
    # merged MODEL row legitimately moved ACTIVE -> PARTIAL because only one
    # of eight upstream embedding memberships is live. Re-pin removes that
    # model row from this lifecycle-scoped runnable population; its completed
    # fold commands remain preserved in embeddings-one-surface.md.
    "SERVE-POOLING-ENDPOINTS",
    "KV-SLIDING-WINDOW-SPEC",
    "LOAD-SAFETENSORS-DIRECT-DENSE",
    "MODEL-FACTORY-registry",
    "MODEL-TEXT-gemma4-gemma4-for-causal-lm",
    "MODEL-TEXT-glm4-glm4-for-causal-lm",
    "MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm",
    "QUANT-GGUF-COMPUTE",
    "QUANT-NVFP4-CT-W4A16",
    "SERVE-ASYNC-LLM",
    "SERVE-STREAM-USAGE",
    "TOOLS-STREAMING-PARSER",
    # 2026-08-30: -BACKEND-TENSTORRENT-GDN leaves the runnable population: the
    # row reached DONE (W1 `34fde3502` + W2 `c85af0aaf` landed fresh-review
    # PASS; the Qwen3_5 family runs production, closing #1715), and a DONE row
    # is outside this lifecycle-scoped population. Re-pinned in the same
    # change that moved the row, as the checker requires. The suites the entry
    # credited (full TT suite, CPU gate, agent-preflight.sh) stay pinned on
    # the still-live rows that bind them, so no command actually left the
    # gated population with the row.
    # (Was: 2026-08-22, +GDN inherited-credit growth when its spec landed.)
    # LTX25-VAE-DEVICE-RESIDENCY (2026-08-25, issue #1451): GROWTH, re-pinned in
    # the same change. `KERNEL-LTX2-VAE` is a NEW row rather than an existing one
    # that changed shape, and it enters the runnable population because its
    # Gates section names two test binaries and a ctest sweep that can fail --
    # `test_diffusion_device_seam` (the residency assertion, a host-device
    # transfer COUNT on a fake accelerator) and `test_ltx2_vae` (the committed
    # decode goldens the ten kernel arms were transcribed from). Neither is a
    # `git diff`, so the row earns the entry rather than being carried by one.
    "KERNEL-LTX2-VAE",
    # 2026-08-23: +BACKEND-TENSTORRENT-QWEN35 enters the runnable population
    # when its spec lands (#1715 wiring row). Same inherited-credit shape as
    # BACKEND-TENSTORRENT-GDN above: the credit is the pre-existing full TT
    # suite, CPU gate, and agent-preflight.sh; the row's own sweep/e2e gates
    # are owed (spec-first).
    "BACKEND-TENSTORRENT-QWEN35",
    # 2026-10-03: +MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2. GROWTH,
    # re-pinned in the change that exposed it. The row became runnable at
    # ae07db609, when its gate gained a command that can genuinely fail:
    # scripts/mm/nano_nemotron_vl_ref.py.
    "MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2",
})


def ratchet_errors(records: list[dict]) -> list[str]:
    """A row may not silently lose its gate command.

    Leaving the gated population is legitimate; losing the command is not. So
    only IDs still PRESENT and still gated can be a regression -- anything else
    is a record edit that must re-pin the baseline in the same change.

    The two are reported as SEPARATE, differently worded errors on purpose: a
    single "the count fell" message would make a broken row and a retired row
    look the same, which is this repo's recorded defect class and the reason
    the baseline is a set.

    This half of the pin sees only DROPS. Growth is caught by the equality
    assertion in tests/scripts/test_check_gate_commands.py, so a row that gains
    a command still re-pins RUNNABLE_BASELINE -- see the note above the set.
    """
    runnable = {item["id"] for item in records if item["verdict"] == "runnable"}
    present = {item["id"] for item in records}
    lost = sorted((RUNNABLE_BASELINE - runnable) & present)
    departed = sorted(RUNNABLE_BASELINE - runnable - present)
    errors = []
    if lost:
        errors.append(
            "these rows still exist and are still gated but no longer name a "
            f"command that can fail: {', '.join(lost)}. Repair the row, never "
            "the baseline."
        )
    if departed:
        errors.append(
            f"these baseline rows left the gated population: {', '.join(departed)}. "
            "If that is a legitimate record edit, re-pin RUNNABLE_BASELINE in the "
            "SAME change, naming each row and the reason."
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify gated rows' gate commands.")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument("--check", action="store_true", help="fail on a ratchet regression")
    args = parser.parse_args(argv)

    # A matrix that did not parse fails EVERY mode, including --json. Reporting
    # a partial audit as if it were the record is how a parse failure ends up
    # wearing the face of a legitimate record edit.
    try:
        records = audit()
    except RecordParseError as exc:
        for line in exc.errors:
            print(f"ERROR: {line}", file=sys.stderr)
        print(
            "ERROR: a matrix did not PARSE, so this audit is incomplete and its "
            "row set means nothing. Repair the matrix. Do NOT re-pin "
            "RUNNABLE_BASELINE off a failed parse.",
            file=sys.stderr,
        )
        return 1
    # BEFORE --json, which returns 0 whatever the record says. If --json won,
    # `--check --json` would be a gate that cannot fail -- the shape this whole
    # file exists to detect, wearing this file's own face.
    if args.check:
        errors = ratchet_errors(records)
        for line in errors:
            print(f"ERROR: {line}", file=sys.stderr)
        return 1 if errors else 0
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    counts: dict[str, int] = {}
    for item in records:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    for verdict in ("runnable", "gates-no-command", "no-gates-section", "no-spec"):
        print(f"  {counts.get(verdict, 0):4d}  {verdict}")
    print(f"\n{len(records)} gated rows; {counts.get('runnable', 0)} carry a command that can fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
