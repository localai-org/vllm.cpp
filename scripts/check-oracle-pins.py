#!/usr/bin/env python3
"""GATE-ORACLE-PINS (#647) — every oracle is named, pinned, and honest about it.

`AGENTS.md` §"When vLLM has no implementation" admits a secondary oracle only
where vLLM implements nothing, only from a fixed set, and only at a recorded
pin. This checker is what stops that rule decaying into prose:

  * every `.agents/oracles/<id>.md` carries exactly one ```oracle-pin block with
    every required key present and non-empty, and no key nobody defined;
  * exactly one record is `role = primary`, and it is `vllm`;
  * the AGENTS.md admissible-oracle table and the directory name the SAME ids,
    in both directions — so an oracle cannot be admitted in prose without a pin
    file, nor pinned in a file nobody is allowed to use;
  * `gateable = yes` requires a revision that is not `UNPINNED` AND evidence
    that is a path EXISTING IN THIS TREE. An issue number is a promise of a
    measurement, so it is refused here and required in the other direction:
    `gateable = no` must name the issue that owes it, which is what keeps an
    ungateable lane visible debt instead of a mid-campaign discovery;
  * the `vllm` record's `pin` and `pin_label` AGREE with the ```parity-pin
    block in `.agents/upstream-sync.md`, which is the authority (#2829).
  * every `pin:commit` / `pin:label` span in the PROSE surfaces named by
    `PIN_SURFACES` agrees with that same authority, and each of those paths
    carries at least one `commit` span (#2883);
  * no marker in those surfaces STARTS ITS LINE, because at column 1 an HTML
    comment is a CommonMark block that interrupts the paragraph and stops the
    span's value rendering as Markdown. See `PIN_AT_LINE_START`.

THE PARITY RECONCILIATION, because it reverses what this docstring used to say.

Until #2829 this file argued that comparing the two surfaces would "BLESS the
duplication, not remove it", and declined to. That reasoning is right about the
FIX and it left the hole open in the meantime: with `upstream-sync.md` advanced
and `oracles/vllm.md` rolled back to the prior revision, this checker,
`tests/tools/test_oracle_pin` and `check-agent-record.py` were all measured
GREEN over a tree whose two pin surfaces named different vLLM commits. A pin
advance is exactly the edit that leaves one surface behind, and the value had
not moved since 2026-07-26, so the drift had never had an opportunity to appear.

So the rule below is the INTERIM, and it says which file is authoritative rather
than syncing two equals. The expectation is read from `upstream-sync.md`; the
value under test is read from `oracles/vllm.md`; an anchor checker that read its
expectation out of the file it checks would be a tautology. Removing the copy
altogether is still the better fix and is still available on top of this: see
`.agents/specs/oracle-pin-parity-reconcile.md` for why it needs a per-id
exemption from `REQUIRED_KEYS` that this change did not take.

It is scoped to `vllm.md` BY FILENAME STEM. Every other oracle holds its pin
only in its own file and has no `parity-pin` counterpart, so a registry-wide
version of this rule would fail all thirteen of them for being correct.

WHAT IT DOES NOT DO, so nobody cites it for more than it delivers:

  * It does not verify that a pin EXISTS upstream. It is deliberately
    network-free — a gate that fails when GitHub is unreachable fails on the
    wrong thing. A fabricated 40-hex string passes shape and fails review.
  * It does not REMOVE the duplication it now reconciles. `oracles/vllm.md`
    still stores a literal, and a checked copy is weaker than no copy.
  * The `**Secondary oracle:**` declaration check is OPT-IN by syntax. A spec
    that merely mentions llama.cpp in prose is not scanned, because a checker
    that fired on the word "SGLang" would go red on hundreds of files that are
    doing nothing wrong and would be silenced within a week.

THE PROSE SURFACES, and why they are delimited rather than parsed (#2883).

`.agents/NOW.md`, `docs/FEATURES.md`, `docs/benchmarks/how-we-measure.md`,
`docs/benchmarks/speculative-decoding.md` and
`docs/benchmarks/vllm-online-serving.md` restate the abbreviated revision and
the release label in sentences. #2880
declined to gate them and its argument was narrower than it read: a grep for a
prior pin returns files that name it ON PURPOSE, which rules out a REGEX OVER
PROSE and does not rule out a rule over NAMED PATHS. #2880 measured 286 such
files; the figure was restated here and in two specs as if measured again, and
it is not the number this tree carries -- `git grep -l 555967922` returns 846
at `aa7056b46`. Either count rules out the regex, which is why the conclusion
stands and the figure did not.

A token rule over unmarked prose was built far enough to reject it, and it fails
on real content in both directions. `docs/FEATURES.md` names the llama.cpp pin
`b10451` one line below the vLLM pin, six characters drawn entirely from
[0-9a-f], so a "backticked hex word" heuristic reads a second project's pin as a
malformed vLLM revision; and a containment rule asking whether the public
version appears in the sentence passes a surface reading `0.28.1rc1.dev1320`.

So the surfaces are DELIMITED instead. An inline `<!--pin:commit-->` /
`<!--pin:label-->` span makes the span content the value: no token regex, no
sentence shape, no positional assumption, and nothing outside a span is read, so
the rule cannot fire when somebody rewraps a paragraph. A marker renders as
nothing ONLY WHERE IT IS INLINE, so `PIN_AT_LINE_START` is a rule and not a
convention: the first version of this change asserted the invisibility in three
documents, measured it in none, and broke a paragraph of `docs/FEATURES.md` on
GitHub's own renderer. The markers' ABSENCE from a named path is an error, which
is what stops the rule being silenced by deleting them.

Run with `--self-test` to sweep the FIXTURES corpus below in both directions:
every `bad=True` fixture must be reported and every `bad=False` one must not.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / ".agents/oracles"
AGENTS_MD = ROOT / "AGENTS.md"
DECLARATION_ROOTS = (ROOT / ".agents",)
# The AUTHORITY for the vLLM revision. `tools/bench/serve_low_common.py` reads
# this same block and `tools/bench/online_gate.py` refuses a mismatched oracle
# on it, so it is what a measurement actually ran against.
UPSTREAM_SYNC = ROOT / ".agents/upstream-sync.md"

# The PROSE surfaces (#2883). The pin is restated in public projections, and
# until now nothing reconciled them against the authority above. They are read
# ONLY inside a `<!--pin:...-->` span, so the hundreds of files that name a
# PRIOR pin on purpose are never looked at, and neither is any unmarked sentence
# in these files. `docs/FEATURES.md` restates the CURRENT pin twice -- line 14
# and again in the *Results.* paragraph -- and BOTH are marked. An unmarked
# restatement of the current pin inside a named surface is exactly the pin
# nothing reconciles that #2883 filed, so leaving one deliberately free would
# have kept half the defect.
# Each of these must exist and carry at least ONE `commit` span, and every span
# in them is validated. The rule reads THESE PATHS AND NOTHING ELSE.
#
# A GLOB OVER `.agents/` AND `docs/` WAS BUILT FIRST AND REJECTED, on evidence.
# It reddened on this rule's OWN SPEC, which quotes the markers in a fenced
# block to explain them: more openers than complete spans, and a "not a
# hexadecimal revision" whose value was half the document. No count is written
# down here on purpose -- it is a measurement of ANOTHER file, it moves whenever
# somebody explains the marker one more time, and this comment carried it wrong
# for exactly that reason. Documentation of a marker is not a surface carrying
# one, and no cheap rule tells them apart. Scanning named paths is also exactly
# what #2883 proposed. The cost is one line here per surface.
PIN_SURFACES = (
    ROOT / ".agents/NOW.md",
    ROOT / "docs/FEATURES.md",
    ROOT / "docs/benchmarks/how-we-measure.md",
    ROOT / "docs/benchmarks/speculative-decoding.md",
    ROOT / "docs/benchmarks/vllm-online-serving.md",
)

REQUIRED_KEYS = (
    "id",
    "role",
    "upstream",
    "scope",
    "pin",
    "pin_label",
    "pinned_on",
    "gateable",
    "evidence",
)

PRIMARY_ID = "vllm"

BLOCK = re.compile(r"^```oracle-pin\n(.*?)^```", re.M | re.S)
PARITY_BLOCK = re.compile(r"^```parity-pin\n(.*?)^```", re.M | re.S)
# Mirrors `tools/bench/serve_low_common.py:_PIN_FIELDS`. Kept as a shape rule
# here and not imported: this checker must run with no `tools/` on the path.
PARITY_FIELDS = (
    "vllm_commit",
    "vllm_runtime_version",
    "vllm_distribution_version",
    "flashinfer_version",
)
FIELD = re.compile(r"^([a-z_]+)\s*=\s*(.*)$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISSUE_REF = re.compile(r"^#\d+$")
# The AGENTS.md table between the registry markers; the `id` column is fenced.
REGISTRY_REGION = re.compile(
    r"<!--\s*oracle-registry:begin\s*-->(.*?)<!--\s*oracle-registry:end\s*-->", re.S
)
TABLE_ID = re.compile(r"^\|[^|]*\|\s*`([a-z0-9-]+)`\s*\|")
DECLARATION = re.compile(r"\*\*Secondary oracle:\*\*\s*`([a-z0-9-]+)`")

# The pin-surface delimiters. `PIN_SPAN` matches a COMPLETE span; `PIN_OPEN`
# matches an opener whose KIND is any identifier, defined or not. Counting the
# two against each other is what turns an unclosed or misspelled marker into an
# error instead of a span that quietly stops existing.
#
# THE OPENER CLASS IS WIDER THAN THE KIND SET ON PURPOSE, and it is not "any
# opener at all", which is what this comment used to claim. It was `[a-z]*`,
# which could not express kinds it was supposed to catch: `pin:LABEL`,
# `pin:Commit`, `pin:commit2` and `pin:pin-commit` matched NEITHER regex, so
# beside one valid span the counts agreed and the misspelled marker was
# reported by nothing -- `pin:LABEL` and `pin:Commit` produced zero errors, and
# the shape reproduces through `main` at rc 0. A detector that cannot represent
# the defect is not a fail-closed rule. `[A-Za-z0-9_-]*` covers every kind
# anybody would plausibly typo; a kind outside it (a space, a non-ASCII letter)
# is still unmatched by both regexes and still passes, which is a smaller hole
# and is named here rather than claimed away.
PIN_SPAN = re.compile(r"<!--\s*pin:(commit|label)\s*-->(.*?)<!--\s*/pin\s*-->", re.S)
PIN_OPEN = re.compile(r"<!--\s*pin:([A-Za-z0-9_-]*)\s*-->")
# A marker that STARTS A LINE, which silently breaks the rendered document.
#
# Under CommonMark an HTML comment indented fewer than four spaces begins a
# type-2 HTML block, and type 2 is one of the seven start conditions that may
# INTERRUPT A PARAGRAPH. The rest of that line is then emitted as raw HTML: the
# span's own value stops being Markdown, one paragraph renders as three blocks,
# and the reader sees literal backticks where a `<code>` was.
#
# MEASURED ON GITHUB'S OWN RENDERER (`gh api /markdown`), one probe per shape,
# because the previous version of this change reasoned about this and was wrong:
#
#   indent 0-3, mid-paragraph or after a blank line -> HTML block, paragraph
#     splits, the value renders as literal backticks;
#   indent >= 4 after a blank line -> INDENTED CODE BLOCK, and the marker
#     itself becomes visible text, which is worse;
#   indent >= 4 (or a tab) inside a paragraph -> a lazy continuation line, and
#     the marker is inline and invisible.
#
# So the rule -- "a pin marker is never first on its line" -- refuses exactly
# one shape that would have rendered correctly: a marker behind four spaces of
# a paragraph CONTINUATION line. Nobody writes that, and it is one blank line
# away from the worst of the three outcomes. The conservative boundary is
# deliberate and is the cost of a rule with no renderer in it.
#
# The defect this catches is not hypothetical. `docs/FEATURES.md:46` shipped a
# column-1 opener inside the *Results.* paragraph: `gh api /markdown` on the
# whole file returned 21 `<p>` for a document with 20 paragraphs and printed
# ``e126687a9a`` with its backticks, and the repaired file returns 20 `<p>` and
# a second `<code>e126687a9a</code>`. No renderer is imported here -- this
# checker is deliberately dependency-free and network-free, and a gate that
# fails when GitHub is unreachable fails on the wrong thing -- so the rule is
# the POSITION that produces the break, which is exactly checkable from bytes.
PIN_AT_LINE_START = re.compile(r"^[ \t]*(<!--\s*(?:pin:[A-Za-z0-9_-]*|/pin)\s*-->)", re.M)
HEX = re.compile(r"^[0-9a-f]+$")
# Hexadecimal in ANY case, so a value that is hex but not lowercase is reported
# for what it is. `vllm_commit` is stored lowercase and the comparison below is
# `str.startswith`, so `E126687A` is a genuine mismatch -- but telling its
# author it "is not a hexadecimal revision" is false about the value they wrote.
HEX_ANY = re.compile(r"^[0-9a-fA-F]+$")
# The FLOOR on an abbreviated revision, and the whole of what it is for.
#
# #2883 asks a rule to settle the abbreviation length, and observes that
# accepting any prefix accepts `e1`. The length is NOT pinned, because this tree
# has no such convention to pin: it writes the prior pin as EIGHT characters in
# `docs/benchmarks/how-we-measure.md`, NINE in `docs/FEATURES.md` and FORTY in
# `.agents/oracles/vllm.md`. A fixed length would invent a convention and go red
# on the shape the repository already uses, and AGENTS.md is explicit that a
# gate which fires on ordinary work is the defect.
#
# Correctness is therefore "is a prefix of vllm_commit". The floor exists for
# ONE reason: below eight hex characters an abbreviation can collide with a
# DIFFERENT commit by accident, and the rule would then pass a stale surface.
# Eight is 4.3e9 values and is the shortest abbreviation this tree writes, so
# the floor rejects `e1` without rejecting anything anybody does today.
MIN_ABBREV = 8


@dataclass
class Record:
    path: Path
    fields: dict[str, str]


def parse_record(path: Path, text: str, errors: list[str]) -> Record | None:
    """Extract the one oracle-pin block, reporting a missing or duplicated one."""
    blocks = BLOCK.findall(text)
    if len(blocks) != 1:
        errors.append(
            f"{path.name}: expected exactly one ```oracle-pin block, found {len(blocks)}"
        )
        return None
    fields: dict[str, str] = {}
    for line_no, line in enumerate(blocks[0].splitlines(), 1):
        if not line.strip():
            continue
        match = FIELD.match(line)
        if match is None:
            errors.append(f"{path.name}: oracle-pin line {line_no} is not `key = value`: {line!r}")
            continue
        key, value = match.group(1), match.group(2).strip()
        if key in fields:
            errors.append(f"{path.name}: duplicate key {key!r} in oracle-pin block")
        fields[key] = value
    return Record(path=path, fields=fields)


def check_record(record: Record, errors: list[str]) -> None:
    """Field-level rules for one record."""
    name = record.path.name
    fields = record.fields

    for key in REQUIRED_KEYS:
        if key not in fields:
            errors.append(f"{name}: oracle-pin is missing required key {key!r}")
        elif not fields[key]:
            errors.append(f"{name}: oracle-pin key {key!r} is empty")
    for key in fields:
        if key not in REQUIRED_KEYS:
            errors.append(f"{name}: oracle-pin carries unknown key {key!r}")

    stem = record.path.stem
    if fields.get("id") and fields["id"] != stem:
        errors.append(f"{name}: id {fields['id']!r} does not match filename stem {stem!r}")

    role = fields.get("role")
    if role and role not in ("primary", "secondary"):
        errors.append(f"{name}: role {role!r} is neither 'primary' nor 'secondary'")

    pinned_on = fields.get("pinned_on")
    if pinned_on and not ISO_DATE.match(pinned_on):
        errors.append(f"{name}: pinned_on {pinned_on!r} is not an ISO date")

    gateable = fields.get("gateable")
    if gateable and gateable not in ("yes", "no"):
        errors.append(f"{name}: gateable {gateable!r} is neither 'yes' nor 'no'")

    evidence = fields.get("evidence", "")
    if gateable == "yes":
        if fields.get("pin") == "UNPINNED":
            errors.append(f"{name}: gateable = yes but pin is UNPINNED — name the revision")
        if ISSUE_REF.match(evidence):
            errors.append(
                f"{name}: gateable = yes cites issue {evidence} as evidence; "
                "an issue is a promise of a measurement, not one"
            )
        elif evidence and not (ROOT / evidence).exists():
            errors.append(f"{name}: evidence path {evidence!r} does not exist in this tree")
    elif gateable == "no" and evidence and not ISSUE_REF.match(evidence):
        errors.append(
            f"{name}: gateable = no must name the owing issue as `#N`, got {evidence!r}"
        )


def parse_parity_pin(label: str, text: str, errors: list[str]) -> dict[str, str] | None:
    """Extract the one ```parity-pin block from *text*, the pin's AUTHORITY.

    FAILS CLOSED on every defect -- no block, two blocks, an unparsable line, an
    unknown key, a duplicate key, an empty value, a missing key. A rule that
    silently stops checking when it cannot find its expectation is a mute
    switch, and this repository has shipped one of those before.
    """
    blocks = PARITY_BLOCK.findall(text)
    if len(blocks) != 1:
        errors.append(
            f"{label}: expected exactly one ```parity-pin block, found {len(blocks)} "
            "-- it is the authority the vllm oracle record is reconciled against"
        )
        return None
    fields: dict[str, str] = {}
    for line_no, line in enumerate(blocks[0].splitlines(), 1):
        if not line.strip():
            continue
        match = FIELD.match(line)
        if match is None:
            errors.append(f"{label}: parity-pin line {line_no} is not `key = value`: {line!r}")
            continue
        key, value = match.group(1), match.group(2).strip()
        if key not in PARITY_FIELDS:
            errors.append(f"{label}: parity-pin carries unknown key {key!r}")
            continue
        if key in fields:
            errors.append(f"{label}: duplicate key {key!r} in parity-pin block")
        if not value:
            errors.append(f"{label}: parity-pin key {key!r} is empty")
            continue
        fields[key] = value
    for key in PARITY_FIELDS:
        if key not in fields:
            errors.append(f"{label}: parity-pin block omits {key!r}")
    return fields


def public_version(version: str) -> str:
    """The PEP 440 public version: everything before the `+local` segment.

    setuptools-scm builds the pinned oracle's version as `<public>+g<sha>`, so
    `0.28.1rc1.dev132+ge126687a9` has the public version `0.28.1rc1.dev132`.
    A build from a tag carries no local segment and `partition` returns the
    whole string, which is the degenerate case this rule wants.
    """
    return version.partition("+")[0]


def check_parity_reconciliation(
    record: Record | None, parity: dict[str, str] | None, errors: list[str]
) -> None:
    """Reconcile the vllm record's `pin`/`pin_label` against the AUTHORITY.

    *parity* comes from `.agents/upstream-sync.md` and *record* from
    `.agents/oracles/vllm.md`. The expectation is a PARAMETER and this function
    opens no file, which is what stops it degenerating into a checker that reads
    its expectation out of the same file it is checking.

    `pin_label` is compared to the PUBLIC part of `vllm_runtime_version`, by
    equality after that decomposition and NOT by prefix. A `startswith` rule
    would accept `0.28.1rc1.dev13`, a truncated and wrong label that is
    nevertheless a prefix of the right one.
    """
    if record is None:
        errors.append(
            f"{PRIMARY_ID}.md: the parity reconciliation could not read the primary "
            "oracle record, so the pin is unchecked against .agents/upstream-sync.md"
        )
        return
    if parity is None:
        return  # parse_parity_pin already reported why, and it reported it once.

    authority = UPSTREAM_SYNC.name
    name = record.path.name
    commit = parity.get("vllm_commit")
    if commit and record.fields.get("pin") != commit:
        errors.append(
            f"{name}: pin {record.fields.get('pin')!r} does not match "
            f"{authority} vllm_commit {commit!r} -- the parity-pin block is the "
            "authority, and this record restates it"
        )
    runtime = parity.get("vllm_runtime_version")
    if runtime:
        expected = public_version(runtime)
        if record.fields.get("pin_label") != expected:
            errors.append(
                f"{name}: pin_label {record.fields.get('pin_label')!r} does not match "
                f"the public version {expected!r} of {authority} "
                f"vllm_runtime_version {runtime!r}"
            )


def surface_label(path: Path) -> str:
    """A path as it is reported: repo-relative when it is inside the tree."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_pin_surfaces(
    surfaces: dict[Path, str],
    required: tuple[Path, ...],
    parity: dict[str, str] | None,
    errors: list[str],
) -> None:
    """The pin's PROSE surfaces agree with the authority (#2883).

    *surfaces* is `{path: text}` and *required* the paths that must each carry
    at least one `commit` span; the two are keyed the same way, and `main`
    passes absolute paths that `surface_label` reports relative to `ROOT`.
    *parity* is the fields read from `.agents/upstream-sync.md`. Like `check_parity_reconciliation`, this
    function opens NO file: the expectation arrives as a parameter, because a
    checker that reads its expectation out of the file it is checking is a
    tautology and this repository has shipped that shape.

    NOTHING OUTSIDE A MARKED SPAN IS READ. That is the whole reason the surfaces
    were structured first rather than parsed. A token rule over an unmarked
    sentence reads `b10451` -- the llama.cpp pin, one line below the vLLM pin in
    `docs/FEATURES.md`, and six characters drawn entirely from [0-9a-f] -- as a
    malformed vLLM revision; a containment rule over the same sentence accepts a
    surface reading `0.28.1rc1.dev1320`. Both failures are unreachable here
    because the span content IS the value.

    The one thing outside a span that IS read is the marker's own POSITION. A
    marker that starts its line changes the RENDERED document, and a rule whose
    whole justification is "this is invisible to a reader" has to be able to
    tell when it is not. See `PIN_AT_LINE_START`.
    """
    if parity is None:
        return  # parse_parity_pin already reported why, and it reported it once.

    commit = parity.get("vllm_commit")
    runtime = parity.get("vllm_runtime_version")
    expected_label = public_version(runtime) if runtime else None

    seen_commit_spans: dict[Path, int] = {}
    for path in sorted(surfaces):
        text = surfaces[path]
        label = surface_label(path)
        spans = PIN_SPAN.findall(text)
        openers = PIN_OPEN.findall(text)
        if len(openers) != len(spans):
            errors.append(
                f"{label}: {len(openers)} `<!--pin:...-->` opener(s) but "
                f"{len(spans)} complete span(s) -- an unclosed or misspelled "
                "marker is an error, not a span that stops existing"
            )
        for match in PIN_AT_LINE_START.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{label}:{line_no}: pin marker {match.group(1)} starts its "
                "line, so CommonMark reads it as a type-2 HTML block that "
                "interrupts the paragraph and emits the rest of the line raw "
                "-- the span's value then renders as literal Markdown. Reflow "
                "so text precedes the marker on its line"
            )
        seen_commit_spans[path] = sum(1 for kind, _ in spans if kind == "commit")
        for kind, raw in spans:
            value = raw.strip().strip("`").strip()
            if not value:
                errors.append(f"{label}: empty `{kind}` pin span")
                continue
            if kind == "commit":
                if commit is None:
                    continue  # parse_parity_pin already reported the omission.
                if not HEX.match(value):
                    if HEX_ANY.match(value):
                        errors.append(
                            f"{label}: pin span {value!r} is hexadecimal but "
                            "not lowercase -- "
                            f"{UPSTREAM_SYNC.name} stores vllm_commit "
                            "lowercase and the prefix comparison is "
                            "case-sensitive"
                        )
                    else:
                        errors.append(
                            f"{label}: pin span {value!r} is not a hexadecimal revision"
                        )
                elif len(value) < MIN_ABBREV:
                    errors.append(
                        f"{label}: pin span {value!r} is {len(value)} characters, "
                        f"under the {MIN_ABBREV}-character floor -- a shorter "
                        "abbreviation can collide with a different commit"
                    )
                elif not commit.startswith(value):
                    errors.append(
                        f"{label}: pin span {value!r} is not a prefix of "
                        f"{UPSTREAM_SYNC.name} vllm_commit {commit!r} -- the "
                        "parity-pin block is the authority, and this surface "
                        "restates it"
                    )
            else:  # kind == "label"; PIN_SPAN admits no third kind.
                if expected_label is None:
                    continue  # parse_parity_pin already reported the omission.
                if value != expected_label:
                    errors.append(
                        f"{label}: pin span {value!r} does not match the public "
                        f"version {expected_label!r} of {UPSTREAM_SYNC.name} "
                        f"vllm_runtime_version {runtime!r}"
                    )

    for path in required:
        label = surface_label(path)
        if path not in surfaces:
            errors.append(
                f"{label}: a declared pin surface, but it is unreadable -- it "
                "restates the pin and must carry a `<!--pin:commit-->` span"
            )
        elif not seen_commit_spans.get(path):
            errors.append(
                f"{label}: carries no `<!--pin:commit-->` span, so the pin it "
                "restates is unchecked against " + UPSTREAM_SYNC.name
            )


def agents_registry_ids(text: str, errors: list[str]) -> list[str]:
    """The `id` column of the admissible-oracle table in AGENTS.md."""
    region = REGISTRY_REGION.search(text)
    if region is None:
        errors.append("AGENTS.md: oracle-registry markers are missing")
        return []
    ids = [m.group(1) for line in region.group(1).splitlines() if (m := TABLE_ID.match(line))]
    if not ids:
        errors.append("AGENTS.md: oracle-registry region names no oracle ids")
    return ids


def check_registry(files: dict[Path, str], agents_ids: list[str]) -> list[str]:
    """Whole-registry rules over `{path: text}` plus the ids AGENTS.md admits."""
    errors: list[str] = []
    records: list[Record] = []
    for path in sorted(files):
        record = parse_record(path, files[path], errors)
        if record is None:
            continue
        check_record(record, errors)
        records.append(record)

    primaries = [r for r in records if r.fields.get("role") == "primary"]
    if len(primaries) != 1:
        errors.append(
            f"registry: expected exactly one role = primary, found {len(primaries)}"
        )
    for primary in primaries:
        if primary.fields.get("id") != PRIMARY_ID:
            errors.append(
                f"{primary.path.name}: the primary oracle is {PRIMARY_ID!r}, "
                f"not {primary.fields.get('id')!r} — a secondary never outranks vLLM"
            )

    # Multiset, not set (CHECKER-HARDENING, FLINT-CPU-PARITY M4): a set diff
    # cannot see DUPLICATION, so a registry row admitted twice by the AGENTS.md
    # table — or a pin id pinned by two records — passed while every set held
    # the same elements. Compare the multiplicities and name each id whose
    # count differs.
    record_ids = [r.fields.get("id", r.path.stem).strip() for r in records]
    record_counts = collections.Counter(record_ids)
    agents_counts = collections.Counter(agents_ids)
    for oracle_id in sorted(set(record_counts) | set(agents_counts)):
        pinned = record_counts[oracle_id]
        admitted = agents_counts[oracle_id]
        if pinned == admitted:
            continue
        if pinned == 0:
            errors.append(
                f"{oracle_id}: admitted by the AGENTS.md table but has no "
                ".agents/oracles/ record, so it has no pin"
            )
        elif admitted == 0:
            errors.append(
                f"{oracle_id}: pinned in .agents/oracles/ but absent from the AGENTS.md table"
            )
        elif admitted > 1:
            errors.append(
                f"{oracle_id}: admitted {admitted}x by the AGENTS.md table but "
                f"pinned {pinned}x — a duplicated registry row"
            )
        else:
            errors.append(
                f"{oracle_id}: pinned {pinned}x in .agents/oracles/ but admitted "
                f"{admitted}x by the AGENTS.md table — a duplicated pin id"
            )
    return errors


def check_declarations(files: dict[Path, str], registry_ids: set[str]) -> list[str]:
    """`**Secondary oracle:** \\`id\\`` must name a registered oracle."""
    errors: list[str] = []
    for path in sorted(files):
        for declared in DECLARATION.findall(files[path]):
            if declared not in registry_ids:
                errors.append(
                    f"{path}: declares secondary oracle `{declared}`, which has no "
                    ".agents/oracles/ record"
                )
    return errors


# Corpus swept by --self-test, in BOTH directions. The bound is honest: it
# covers the rules below and nothing else, so a rule added without a fixture is
# covered only by the unit tests in tests/scripts/test_check_oracle_pins.py.
_GOOD = """```oracle-pin
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

FIXTURES: tuple[tuple[str, str, bool], ...] = (
    ("fixture.md", _GOOD, False),
    ("fixture.md", _GOOD.replace("pinned_on = 2026-08-13\n", ""), True),
    ("fixture.md", _GOOD.replace("scope = a path vLLM does not implement", "scope ="), True),
    ("fixture.md", _GOOD.replace("gateable = no", "gateable = someday"), True),
    ("fixture.md", _GOOD.replace("evidence = #647", "evidence = soon"), True),
    ("fixture.md", _GOOD.replace("pinned_on = 2026-08-13", "pinned_on = august"), True),
    ("fixture.md", _GOOD.replace("role = secondary", "role = advisory"), True),
    ("fixture.md", _GOOD + _GOOD, True),
    ("renamed.md", _GOOD, True),
    (
        "fixture.md",
        _GOOD.replace("gateable = no\nevidence = #647", "gateable = yes\nevidence = #647"),
        True,
    ),
    (
        "fixture.md",
        _GOOD.replace(
            "gateable = no\nevidence = #647", "gateable = yes\nevidence = AGENTS.md"
        ),
        True,  # gateable = yes with pin = UNPINNED
    ),
)


# The parity reconciliation's own corpus, swept in both directions beside the
# one above. It is separate because the rule is CROSS-FILE: every case pairs a
# synthetic authority with a synthetic record, so none of it moves when the real
# pin advances.
_PARITY_RECORD = """```oracle-pin
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

_PARITY_SYNC = """```parity-pin
vllm_commit = 1111111111111111111111111111111111111111
vllm_runtime_version = 9.9.9rc1.dev1+g1111111111
vllm_distribution_version = 9.9.9rc1.dev1+g1111111111.precompiled
flashinfer_version = 0.6.18
```
"""

# The registry multiplicity corpus (CHECKER-HARDENING), swept beside the record
# corpus. The rule is a MULTISET comparison: a duplicated AGENTS.md registry row
# must fail exactly as an unregistered id does, which a set diff cannot see.
_REGISTRY_PRIMARY = {
    "vllm": """# fixture primary

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
""",
}
_REGISTRY_SECONDARY = {
    "fixture": """# fixture secondary

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
""",
}

REGISTRY_FIXTURES: tuple[tuple[str, dict[str, str], list[str], bool], ...] = (
    ("agree", {**_REGISTRY_PRIMARY, **_REGISTRY_SECONDARY}, ["vllm", "fixture"], False),
    # The M4 shape: the SAME row admitted twice. A set comparison passes this.
    (
        "duplicated AGENTS registry row",
        {**_REGISTRY_PRIMARY, **_REGISTRY_SECONDARY},
        ["vllm", "fixture", "fixture"],
        True,
    ),
    ("agents entry absent from the registry", {**_REGISTRY_PRIMARY, **_REGISTRY_SECONDARY}, ["vllm", "fixture", "ghost"], True),
    ("registry entry absent from AGENTS", {**_REGISTRY_PRIMARY, **_REGISTRY_SECONDARY}, ["vllm"], True),
)

PARITY_FIXTURES: tuple[tuple[str, str, str, bool], ...] = (
    ("agree", _PARITY_RECORD, _PARITY_SYNC, False),
    # The #2829 shape: the authority advanced and the record did not.
    (
        "stale pin",
        _PARITY_RECORD.replace(
            "pin = 1111111111111111111111111111111111111111",
            "pin = 2222222222222222222222222222222222222222",
        ),
        _PARITY_SYNC,
        True,
    ),
    ("stale label", _PARITY_RECORD.replace("9.9.9rc1.dev1", "9.8.0"), _PARITY_SYNC, True),
    # A PROPER PREFIX of the right label. A substring rule accepts this one.
    (
        "truncated label",
        _PARITY_RECORD.replace("9.9.9rc1.dev1", "9.9.9rc1.dev"),
        _PARITY_SYNC,
        True,
    ),
    # The full runtime string, local segment included, is NOT the label.
    (
        "label carries the local segment",
        _PARITY_RECORD.replace(
            "pin_label = 9.9.9rc1.dev1", "pin_label = 9.9.9rc1.dev1+g1111111111"
        ),
        _PARITY_SYNC,
        True,
    ),
    # A tagged build reports a bare public version, and the rule degrades to
    # plain equality rather than going red on a correct authority.
    (
        "authority with no local segment",
        _PARITY_RECORD,
        _PARITY_SYNC.replace(
            "vllm_runtime_version = 9.9.9rc1.dev1+g1111111111",
            "vllm_runtime_version = 9.9.9rc1.dev1",
        ),
        False,
    ),
    ("no parity block", _PARITY_RECORD, "# nothing here\n", True),
    ("two parity blocks", _PARITY_RECORD, _PARITY_SYNC + _PARITY_SYNC, True),
    (
        "unparsable parity line",
        _PARITY_RECORD,
        _PARITY_SYNC.replace("flashinfer_version = 0.6.18", "flashinfer 0.6.18"),
        True,
    ),
    (
        "unknown parity key",
        _PARITY_RECORD,
        _PARITY_SYNC.replace("flashinfer_version = 0.6.18", "torch_version = 2.13.0"),
        True,
    ),
    (
        "empty parity value",
        _PARITY_RECORD,
        _PARITY_SYNC.replace(
            "vllm_commit = 1111111111111111111111111111111111111111", "vllm_commit ="
        ),
        True,
    ),
    ("no record", "# no oracle-pin block\n", _PARITY_SYNC, True),
)


# The pin-surface corpus (#2883), swept in both directions beside the two above.
# `_PIN_GOOD` is a whole sentence, not a bare span, because the rule's central
# claim is that the prose AROUND a span is free: the benign fixtures rewrap it,
# add a second project's pin (`b10451`, six characters of [0-9a-f], the exact
# token that killed the prose-parsing design) and name the PRIOR pin on purpose.
_PIN_GOOD = (
    "**Oracle pin.** vLLM <!--pin:label-->9.9.9rc1.dev1<!--/pin--> "
    "(<!--pin:commit-->`1111111111`<!--/pin-->) since 2026-09-03.\n"
)

PIN_SURFACE_FIXTURES: tuple[tuple[str, str, bool], ...] = (
    ("agrees", _PIN_GOOD, False),
    # The prose around the spans is free. These three are the "a gate that fires
    # on correct work is the defect" cases, and they are fixtures, not comments.
    (
        "benign rewrap and neighbouring prose",
        "Reference versions: llama.cpp `b10451`, MLX-LM as of 2026-07.\n"
        "Rows were read at the PRIOR pin `555967922` unless they say\n"
        "otherwise.\n\n" + _PIN_GOOD.replace(") since", ")\nsince"),
        False,
    ),
    ("a full 40-character revision", _PIN_GOOD.replace("1111111111", "1" * 40), False),
    ("an 8-character revision, the floor", _PIN_GOOD.replace("1111111111", "11111111"), False),
    ("two commit spans that both agree", _PIN_GOOD + _PIN_GOOD, False),
    # The #2829 shape, in prose this time: the authority advanced, the surface
    # did not.
    ("a stale revision", _PIN_GOOD.replace("1111111111", "2222222222"), True),
    ("a stale label", _PIN_GOOD.replace("9.9.9rc1.dev1<", "9.8.0<"), True),
    # A PROPER PREFIX of the right label, which a substring rule accepts.
    ("a truncated label", _PIN_GOOD.replace("9.9.9rc1.dev1<", "9.9.9rc1.dev<"), True),
    # And the other direction: the right label with a character appended, which
    # a CONTAINMENT rule accepts because the expectation is a substring of it.
    ("a label with a suffix", _PIN_GOOD.replace("9.9.9rc1.dev1<", "9.9.9rc1.dev1320<"), True),
    (
        "a label carrying the local segment",
        _PIN_GOOD.replace("9.9.9rc1.dev1<", "9.9.9rc1.dev1+g1111111111<"),
        True,
    ),
    # The degenerate prefix #2883 names by hand.
    ("a two-character abbreviation", _PIN_GOOD.replace("`1111111111`", "`11`"), True),
    ("a seven-character abbreviation", _PIN_GOOD.replace("`1111111111`", "`1111111`"), True),
    ("a non-hexadecimal revision", _PIN_GOOD.replace("1111111111", "zzzzzzzzzz"), True),
    ("an empty span", _PIN_GOOD.replace("`1111111111`", ""), True),
    ("an unclosed marker", _PIN_GOOD.replace("`<!--/pin-->) since", "`) since"), True),
    ("a marker kind nobody defined", _PIN_GOOD.replace("pin:commit", "pin:sha"), True),
    # The same defect OUTSIDE `[a-z]`, which the opener class could not express
    # until this repair and which therefore passed beside a valid span.
    ("a mis-cased marker kind", _PIN_GOOD.replace("pin:label", "pin:LABEL"), True),
    ("a marker kind with a digit", _PIN_GOOD.replace("pin:commit", "pin:commit2"), True),
    ("a hyphenated marker kind", _PIN_GOOD.replace("pin:commit", "pin:pin-commit"), True),
    # Hexadecimal, and still wrong: the authority is lowercase and the
    # comparison is `startswith`. The message has to say which of the two it is.
    ("an uppercase revision", _PIN_GOOD.replace("1111111111", "AAAAAAAAAA"), True),
    # THE RENDERING CASE. A marker that starts its line is a CommonMark type-2
    # HTML block, it interrupts the paragraph, and the span's value renders as
    # literal Markdown. `docs/FEATURES.md:46` shipped exactly this shape.
    (
        "an opening marker at the start of a line",
        _PIN_GOOD.replace(" (<!--pin:commit-->", "\n<!--pin:commit-->"),
        True,
    ),
    (
        "a closing marker at the start of a line",
        _PIN_GOOD.replace("`1111111111`<!--/pin-->", "`1111111111`\n<!--/pin-->"),
        True,
    ),
    (
        "an indented marker, which is a visible code block at a block boundary",
        _PIN_GOOD.replace(" (<!--pin:commit-->", "\n\n      <!--pin:commit-->"),
        True,
    ),
    # And the benign direction: a marker that merely FOLLOWS text on its line,
    # however little, is inline and renders as nothing.
    (
        "a marker one character into its line",
        _PIN_GOOD.replace(" (<!--pin:commit-->", "\n(<!--pin:commit-->"),
        False,
    ),
    ("no marker at all", "**Oracle pin.** vLLM 9.9.9rc1.dev1 (`1111111111`).\n", True),
    (
        "only a label span, so the revision is unchecked",
        "**Oracle pin.** vLLM <!--pin:label-->9.9.9rc1.dev1<!--/pin-->.\n",
        True,
    ),
)


def pin_surface_errors(surface_text: str) -> list[str]:
    """Only the pin-surface rule's complaints, for one synthetic surface."""
    errors: list[str] = []
    parity = parse_parity_pin(UPSTREAM_SYNC.name, _PARITY_SYNC, errors)
    path = Path("NOW.md")
    check_pin_surfaces({path: surface_text}, (path,), parity, errors)
    return errors


def parity_errors(record_text: str, sync_text: str) -> list[str]:
    """Only the reconciliation's own complaints, for one synthetic pair."""
    errors: list[str] = []
    record = parse_record(Path(f"{PRIMARY_ID}.md"), record_text, [])
    parity = parse_parity_pin(UPSTREAM_SYNC.name, sync_text, errors)
    check_parity_reconciliation(record, parity, errors)
    return errors


def self_test() -> int:
    failures: list[str] = []
    for name, surface_text, bad in PIN_SURFACE_FIXTURES:
        errors = pin_surface_errors(surface_text)
        reported = bool(errors)
        if reported != bad:
            verb = "was not reported" if bad else "was reported"
            failures.append(f"pin-surface fixture {name!r} {verb}: {errors}")
    for name, record_text, sync_text, bad in PARITY_FIXTURES:
        errors = parity_errors(record_text, sync_text)
        reported = bool(errors)
        if reported != bad:
            verb = "was not reported" if bad else "was reported"
            failures.append(f"parity fixture {name!r} {verb}: {errors}")
    for name, files, agents_ids, bad in REGISTRY_FIXTURES:
        registry_errors = check_registry(
            {Path(f"{stem}.md"): text for stem, text in files.items()},
            agents_ids,
        )
        reported = bool(registry_errors)
        if reported != bad:
            verb = "was reported" if reported else "was not reported"
            failures.append(f"registry fixture {name!r} {verb}: {registry_errors}")
    for name, text, bad in FIXTURES:
        errors: list[str] = []
        record = parse_record(Path(name), text, errors)
        if record is not None:
            check_record(record, errors)
        reported = bool(errors)
        if reported != bad:
            verb = "was not reported" if bad else "was reported"
            failures.append(f"fixture {name} {verb}: {errors}")
    for failure in failures:
        print(f"self-test: {failure}", file=sys.stderr)
    total = (
        len(FIXTURES)
        + len(REGISTRY_FIXTURES)
        + len(PARITY_FIXTURES)
        + len(PIN_SURFACE_FIXTURES)
    )
    if failures:
        print(f"self-test FAILED ({len(failures)} of {total})", file=sys.stderr)
        return 1
    print(
        f"self-test ok ({len(FIXTURES)} record fixtures, "
        f"{len(REGISTRY_FIXTURES)} registry fixtures, "
        f"{len(PARITY_FIXTURES)} parity fixtures, "
        f"{len(PIN_SURFACE_FIXTURES)} pin-surface fixtures)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="sweep the fixture corpus")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    if not ORACLES.is_dir():
        print(f"{ORACLES.relative_to(ROOT)}: oracle registry directory is missing", file=sys.stderr)
        return 1

    files = {
        path: path.read_text(encoding="utf-8")
        for path in sorted(ORACLES.glob("*.md"))
        if path.name != "README.md"
    }
    if not files:
        print(f"{ORACLES.relative_to(ROOT)}: registry holds no oracle records", file=sys.stderr)
        return 1

    agents_text = AGENTS_MD.read_text(encoding="utf-8")
    errors: list[str] = []
    agents_ids = agents_registry_ids(agents_text, errors)
    errors.extend(check_registry(files, agents_ids))

    # The parity reconciliation (#2829). Scoped to the primary record BY
    # FILENAME STEM: no other oracle has an external authority, and a
    # registry-wide version of this rule would fail every one of them.
    primary_path = ORACLES / f"{PRIMARY_ID}.md"
    # check_registry already reported any parse defect in this file; a second
    # parse into a scratch list is how the record is obtained without printing
    # the same complaint twice.
    primary = (
        parse_record(primary_path, files[primary_path], [])
        if primary_path in files
        else None
    )
    try:
        sync_text = UPSTREAM_SYNC.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"{UPSTREAM_SYNC.name}: the parity pin authority is unreadable: {error}")
        parity = None
    else:
        parity = parse_parity_pin(UPSTREAM_SYNC.name, sync_text, errors)
    check_parity_reconciliation(primary, parity, errors)

    registry_ids = {path.stem for path in files}
    declarations = {
        path.relative_to(ROOT): path.read_text(encoding="utf-8")
        for root in DECLARATION_ROOTS
        for path in sorted(root.rglob("*.md"))
    }
    errors.extend(check_declarations(declarations, registry_ids))

    # The pin's PROSE surfaces (#2883). Read from the module-level tuple so the
    # whole rule can be pointed at a synthetic tree by the through-`main` tests,
    # which is what proves this call site exists.
    pin_surfaces: dict[Path, str] = {}
    for path in PIN_SURFACES:
        try:
            pin_surfaces[path] = path.read_text(encoding="utf-8")
        except OSError:
            continue  # check_pin_surfaces reports the absence, with the reason.
    check_pin_surfaces(pin_surfaces, PIN_SURFACES, parity, errors)

    for error in errors:
        print(f"oracle-pins: {error}", file=sys.stderr)
    if errors:
        print(
            f"oracle-pins FAILED ({len(errors)} error(s)) — see AGENTS.md "
            '"When vLLM has no implementation"',
            file=sys.stderr,
        )
        return 1
    print(f"oracle-pins ok ({len(files)} oracles pinned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
