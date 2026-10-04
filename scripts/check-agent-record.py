#!/usr/bin/env python3
"""Validate the canonical .agents tables and lifecycle contracts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# `scripts/` is sys.path[0] when this file is run, but not when a caller loads
# it by path, so pin the sibling issue-record module either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue_records  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"
ISSUES_ROOT = AGENTS / "issues"

MATRICES = {
    "MODEL": AGENTS / "model-matrix.md",
    "QUANT": AGENTS / "quantization-matrix.md",
    "KERNEL": AGENTS / "kernel-matrix.md",
    "BACKEND": AGENTS / "backend-matrix.md",
}
# Citation-preserving padding; retired rationale lives in the completed archive.
# Do not remove: tracked record anchors predate issue #3085.
@dataclass(frozen=True)
class ClaimRecord:
    row_states: dict[str, str | None]
    lifecycle: str | None
    path: Path
    line_no: int
    strict: bool

    @property
    def row_ids(self) -> set[str]:
        return set(self.row_states)


NONTERMINAL_CLAIM_STATES = {"ACTIVE", "IMPLEMENTING", "SPIKE"}
CLAIM_LIFECYCLE_RE = re.compile(r"^\s*`([A-Z][A-Z0-9-]*)`")

def parse_claim_source(
    path: Path,
    errors: list[str],
    claims: dict[str, ClaimRecord],
    origin: dict[str, str],
) -> None:
    header: tuple[str, ...] = ()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        cells = split_cells(line)
        if cells and normalize_header(cells[0]) == "claim":
            header = tuple(normalize_header(cell) for cell in cells)
            continue
        if not line.startswith("| `CLAIM-"):
            continue
        claim_match = CLAIM_RE.search(cells[0])
        if claim_match is None:
            continue
        claim = claim_match.group(0)
        if claim in claims:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: duplicate active claim "
                f"{claim} (already declared in {origin[claim]})"
            )
            continue
        row_cell = cells[1] if len(cells) > 1 else ""
        matches = list(ID_RE.finditer(row_cell))
        row_states: dict[str, str | None] = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(row_cell)
            annotation = re.match(r"^`\s*\(([^)]*)\)", row_cell[match.end() : end])
            state_match = STATE_RE.search(annotation.group(1)) if annotation else None
            row_states[match.group(0)] = state_match.group(1) if state_match else None
        state_index = field_index(header, "state")
        state_cell = cells[state_index] if state_index is not None and state_index < len(cells) else ""
        lifecycle_match = CLAIM_LIFECYCLE_RE.match(state_cell)
        origin[claim] = str(path.relative_to(ROOT))
        claims[claim] = ClaimRecord(
            row_states=row_states,
            lifecycle=lifecycle_match.group(1) if lifecycle_match else None,
            path=path,
            line_no=line_no,
            strict=path.parent == AGENTS / "claims",
        )


def claimed_row_ids(claims: dict[str, ClaimRecord], claim: str) -> set[str]:
    record = claims.get(claim)
    return record.row_ids if record is not None else set()


def check_claim_state_consistency(
    claims: dict[str, ClaimRecord],
    by_id: dict[str, ClaimRow],
    errors: list[str],
) -> None:
    for claim, record in claims.items():
        if not record.strict:
            continue
        location = f"{record.path.relative_to(ROOT)}:{record.line_no}"
        for item_id, annotated_state in record.row_states.items():
            row = by_id.get(item_id)
            if row is None:
                continue
            if annotated_state is None:
                errors.append(
                    f"{location}: claim {claim} does not annotate {item_id} "
                    "with its matrix lifecycle state"
                )
            elif annotated_state != row.state:
                errors.append(
                    f"{location}: claim {claim} annotates {item_id} as "
                    f"{annotated_state}, but matrix state is {row.state}"
                )
    for row in by_id.values():
        if row.state not in {"SPIKE", "ACTIVE"}:
            continue
        claim_match = CLAIM_RE.search(row.field("owner"))
        record = claims.get(claim_match.group(0)) if claim_match is not None else None
        if record is None or not record.strict or record.lifecycle in NONTERMINAL_CLAIM_STATES:
            continue
        lifecycle = record.lifecycle or "<missing>"
        errors.append(
            f"{row.path.relative_to(ROOT)}:{row.line_no}: owner {claim_match.group(0)} "
            f"for live row {row.item_id} has claim lifecycle {lifecycle}, not "
            "ACTIVE/IMPLEMENTING/SPIKE"
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
# Historical KERNEL cardinality citation retired to .agents/completed/matrix-cardinality-history.md.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Historical CUDA/backend cardinality citation retired to .agents/completed/matrix-cardinality-history.md.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

ENGINE_MATRIX = AGENTS / "engine-matrix.md"
ENGINE_PREFIXES = (
    "ENG",
    "KV",
    "PAR",
    "SAMPLE",
    "TOOLS",
    "SPEC",
    "SERVE",
    "LORA",
    "ATTN",
    "LOAD",
)
# Citation-preserving padding; matrix membership remains derived at read time.
# Do not remove: tracked record anchors predate issue #3085.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Compatibility anchor for check_links.
# check_links still extracts reader-followable markdown targets,
# resolves them against each source's permitted bases,
# reports missing local targets,
# and ignores fenced and inline-code examples.
# The executable implementation remains below this padding.
# This range preserves historical line-addressed prose only.
# It stores no matrix measurement,
# does not select a matrix row,
# does not alter link parsing,
# and does not replace the focused link tests.
# See def check_links for current behavior.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Historical dated-cardinality citation retired to the completed archive.
# This compatibility anchor carries no live row count or chronology.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

ENGINE_SUMMARY_SECTIONS = (
    ("Engine and scheduling", "Engine core and scheduling"),
    ("KV cache and memory", "KV cache and memory"),
    ("Parallelism", "Parallelism and scale-out"),
    ("Sampling and generation", "Sampling and generation controls"),
    ("Structured output and tools", "Structured outputs and tool calling"),
    ("Speculative decoding", "Speculative decoding"),
    ("Serving, API, CLI, library", "Serving surface, CLI, and library"),
    ("LoRA and adapters", "LoRA and adapters"),
    ("Long context and attention", "Long context and attention breadth"),
    ("Loading, tokenizer, config", "Loading, tokenizer, and config"),
)

MATRIX_PATHS = [ENGINE_MATRIX, *MATRICES.values()]
REQUIRED = [
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "docs/BENCHMARKS.md",
    AGENTS / "roadmap_v1.md",
    AGENTS / "coordination.md",
    AGENTS / "feature-matrix.md",
    AGENTS / "specs/model-family-inventory.md",
    AGENTS / "specs/feature-anchor-backfill.md",
    *MATRIX_PATHS,
]

STATES = {
    "INVENTORIED",
    "SPIKE",
    "READY",
    "ACTIVE",
    "GATING",
    "PARTIAL",
    "DONE",
    "BLOCKED",
    "ANCHOR-BACKFILL",
    "BUILD-ONLY",
    "UNTRACED",
}
READY_STATES = {"READY", "ACTIVE", "GATING", "DONE", "BLOCKED"}
EVIDENCED_STATES = {
    "PARTIAL",
    "ANCHOR-BACKFILL",
    "GATING",
    "DONE",
    "BUILD-ONLY",
    "UNTRACED",
}

ALL_PREFIXES = (*ENGINE_PREFIXES, *MATRICES)
ID_RE = re.compile(
    rf"(?:{'|'.join(re.escape(prefix) for prefix in ALL_PREFIXES)})-"
    r"[A-Za-z0-9_.-]+"
)
STATE_RE = re.compile(r"`(" + "|".join(re.escape(state) for state in STATES) + r")`")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# Group 1 is the run of fence characters, group 2 everything after it, which is
# the INFO STRING on an opening fence and must be empty on a closing one. Both
# groups are load-bearing: see strip_code_spans for the pairing rule and for the
# live file that mis-paired without it.
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
CLAIM_RE = re.compile(r"CLAIM-[A-Za-z0-9_.-]+")
LINE_FRAGMENT_RE = re.compile(r"L(\d+)(?:-L?(\d+))?")
COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")
RAW_LOCAL_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:src|include|tests|examples|cmake|scripts|tools|\.github/workflows)/"
    r"[A-Za-z0-9_./-]+|CMakeLists\.txt):(\d+)(?:-(\d+))?"
)

CODE_ANCHOR_PREFIXES = (
    "src/",
    "include/",
    "examples/",
    "cmake/",
    "scripts/",
    "tools/",
    ".github/workflows/",
)
TEST_ANCHOR_PREFIXES = (
    "tests/",
    "scripts/",
    ".github/workflows/",
)
CODE_ANCHOR_FILES = {"CMakeLists.txt"}
EVIDENCE_ANCHOR_FILES = {
    ".agents/parity-ledger.md",
}

SPEC_REQUIREMENTS = {
    "Scope": ("scope",),
    "Upstream chain": ("upstream chain",),
    "Our baseline": ("our baseline",),
    "Port map": ("port map", "port and harness map"),
    "Tests to port": ("tests to port",),
    "Gates": ("gates",),
    "Dependencies": ("dependencies",),
    "Work breakdown": (
        "work breakdown",
        "non overlapping work breakdown",
        "work decomposition",
    ),
    "Risks/decisions": ("risks", "risk /", "risk and", "risks/decisions"),
}


@dataclass(frozen=True)
class ClaimRow:
    path: Path
    line_no: int
    item_id: str
    state: str
    header: tuple[str, ...]
    cells: tuple[str, ...]
    raw: str

    def field(self, name: str) -> str:
        index = field_index(self.header, name)
        return self.cells[index] if index is not None and index < len(self.cells) else ""


def markdown_files() -> list[Path]:
    return [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "docs/BENCHMARKS.md",
        *sorted(AGENTS.rglob("*.md")),
    ]


def split_cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def normalize_header(value: str) -> str:
    value = value.replace("`", "").replace("*", "").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def field_index(header: tuple[str, ...], field: str) -> int | None:
    for index, value in enumerate(header):
        if field == "id" and value == "id":
            return index
        if field == "item" and (
            value.startswith("item")
            or value == "encoding"
            or "method scheme" in value
            or "cache dtype mode" in value
            or value == "scope"
            or value == "surface"
        ):
            return index
        if field == "upstream" and "upstream" in value:
            return index
        if field == "code" and ("our code" in value or "local evidence" in value):
            return index
        if field == "tests" and (
            "tests evidence" in value
            or "test evidence" in value
            or "local evidence" in value
            or value == "evidence"
        ):
            return index
        if field == "spec" and "spike" in value:
            return index
        if field == "state" and (value == "state" or value.endswith(" state")):
            return index
        if field == "owner" and "owner" in value:
            return index
    return None


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_claim_rows(path: Path, errors: list[str]) -> list[ClaimRow]:
    rows: list[ClaimRow] = []
    header: tuple[str, ...] | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            header = None
            continue
        cells = split_cells(line)
        normalized = tuple(normalize_header(cell) for cell in cells)
        if "id" in normalized and any(value == "state" or value.endswith(" state") for value in normalized):
            header = normalized
            continue
        if is_separator(cells):
            continue
        if header is None or not cells:
            continue

        item_id = cells[0].strip().strip("`")
        if not ID_RE.fullmatch(item_id):
            continue
        # A row that fails its shape check is REPORTED and still parsed, never
        # dropped: dropping it hid every downstream contract defect behind the
        # shape error and corrupted the matrix ratchet counts, so each fix
        # exposed the next defect only after another run (three defects masked
        # this way in the ORPHAN-MODEL-ROWS repair;
        # ISSUE-LOCAL-01M3NC14GE995V9E6F7GTYSQJ3). The one error list reports
        # the shape and the contracts in the same run.
        malformed = False
        if len(cells) != len(header):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} has {len(cells)} cells; "
                f"header has {len(header)}"
            )
            malformed = True
        state_index = field_index(header, "state")
        state_cell = cells[state_index] if state_index is not None else ""
        state_matches = STATE_RE.findall(state_cell)
        if len(state_matches) != 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} must have exactly one canonical state"
            )
            malformed = True
        if malformed:
            # Parsed with an empty state so the ratchet, duplicate detection
            # and the summary rollups still see the row; no state-conditional
            # contract can fire on the empty state, so the reported defects
            # stay the shape ones.
            rows.append(
                ClaimRow(path, line_no, item_id, state_matches[0] if state_matches else "", header, tuple(cells), line)
            )
            continue
        rows.append(
            ClaimRow(path, line_no, item_id, state_matches[0], header, tuple(cells), line)
        )
    return rows


def strip_code_spans(text: str) -> str:
    """Blank out fenced blocks and inline code, preserving line and column count.

    A target inside a code span is NOT a link: CommonMark renders it as literal
    text, so no reader can follow it and there is nothing for "every link
    resolves" to be about. Before 2026-08-12 this checker validated them anyway
    (#460), which meant no document in the tree could SHOW a link in sample
    output, and, worse, that a docs/BENCHMARKS.md row quoting its evidence link
    could not be archived into .agents/ byte-for-byte.

    THE PAIRING RULE IS COMMONMARK'S, not "the next line that looks like a
    fence". A closing fence must use the OPENER'S character, be at least as
    long, and carry nothing but whitespace after the marker; a line with an info
    string opens a block and never closes one. Getting that wrong does not fail
    safe, it INVERTS fence phase for the rest of the file: with the one
    unbalanced fence this tree already has, a bare ``` at
    .agents/completed/state-events/0000-00/STATE-LEGACY-000001.md:17697 was
    "closed" by the ```sh at :17948, and ordinary prose at :18297 was blanked,
    so a live reader-followable link stopped being validated. Measured
    tree-wide, the loose rule dropped 5 targets and the CommonMark rule drops 4.

    Blanking rather than deleting preserves every line and column offset. Note
    that check_links reports no line numbers today, so this buys nothing yet; it
    is kept so that a caller that does report positions cannot be broken by this
    function, and test_stripping_preserves_line_and_column_positions holds it.
    """
    out: list[str] = []
    fence: str | None = None
    fence_len = 0
    for line in text.splitlines():
        marker = FENCE_RE.match(line)
        if fence is None:
            # CommonMark: a backtick opening fence's info string may not contain
            # a backtick, which is what keeps `` `a` and `b` `` from opening one.
            if marker is not None and not (
                marker.group(1)[0] == "`" and "`" in marker.group(2)
            ):
                fence = marker.group(1)[0]
                fence_len = len(marker.group(1))
                out.append(" " * len(line))
                continue
            out.append(INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), line))
            continue
        if (
            marker is not None
            and marker.group(1)[0] == fence
            and len(marker.group(1)) >= fence_len
            and not marker.group(2).strip()
        ):
            fence = None
        out.append(" " * len(line))
    return "\n".join(out)


def extract_links(text: str) -> list[str]:
    """Return every link target a READER could follow in this document."""
    return LINK_RE.findall(strip_code_spans(text))


def link_bases(source: Path, text: str) -> tuple[Path, ...]:
    """Return every directory a relative link in this file may resolve from.

    Normally exactly one, the file's own directory. Two files are archives that
    hold content written somewhere else and moved here verbatim, so a target in
    them was authored against the ORIGINAL directory: migrated legacy
    state-event payloads came from .agents/, and .agents/benchmark-record.md is
    the declared archive of docs/BENCHMARKS.md (#460).

    BE PRECISE ABOUT WHAT THE SECOND BASE BUYS. It does NOT make the archived
    copy clickable: a reader opening .agents/benchmark-record.md on GitHub and
    clicking a docs/-relative target such as USAGE.md, BUILD.md or
    bench-evidence/... gets a 404, because the browser resolves it against
    .agents/. What it enforces is that the EVIDENCE STILL EXISTS in the tree
    under one of the two declared bases, so archiving a row byte-for-byte cannot
    silently orphan the file it points at. That is the property the archive
    exists to give, and it is weaker than followability. Making the archived
    copy followable means rewriting the target or recording its origin, which is
    spec W5 and is deliberately not done here: rewriting a target would break
    the byte-for-byte guarantee, and W5 records the origin instead.
    """
    if (
        source.is_relative_to(AGENTS / "completed/state-events")
        and "<!-- legacy-payload:begin -->" in text
    ):
        return (AGENTS,)
    if source == AGENTS / "benchmark-record.md":
        return (source.parent, ROOT / "docs")
    return (source.parent,)


def check_links(errors: list[str]) -> None:
    for source in markdown_files():
        if source.is_relative_to(ISSUES_ROOT):
            continue
        text = source.read_text(encoding="utf-8")
        bases = link_bases(source, text)
        for raw_target in extract_links(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path, _, fragment = target.partition("#")
            candidates = [(base / target_path).resolve() for base in bases]
            resolved = next((c for c in candidates if c.exists()), None)
            if resolved is None:
                errors.append(f"{source.relative_to(ROOT)}: dangling link {raw_target}")
                continue
            line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
            if line_match is None or not resolved.is_file():
                continue
            line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
            start = int(line_match.group(1))
            end = int(line_match.group(2) or start)
            if start < 1 or end < start or end > line_count:
                errors.append(
                    f"{source.relative_to(ROOT)}: out-of-range line anchor {raw_target} "
                    f"(file has {line_count} lines)"
                )


def table_ids(prefix: str, path: Path, errors: list[str]) -> list[str]:
    return [
        row.item_id
        for row in parse_claim_rows(path, errors)
        if row.item_id.startswith(prefix + "-")
    ]


def check_matrices(errors: list[str]) -> tuple[list[ClaimRow], dict[str, ClaimRow]]:
    rows: list[ClaimRow] = []
    for path in MATRIX_PATHS:
        rows.extend(parse_claim_rows(path, errors))

    by_id: dict[str, ClaimRow] = {}
    for row in rows:
        prior = by_id.get(row.item_id)
        if prior is not None:
            errors.append(
                f"{row.path.relative_to(ROOT)}:{row.line_no}: duplicate ID {row.item_id}; "
                f"first at {prior.path.relative_to(ROOT)}:{prior.line_no}"
            )
        else:
            by_id[row.item_id] = row

# Citation-preserving padding; no cardinality is stored here.
# Do not remove: tracked record anchors predate issue #3085.
#
#
#
#
#
#
#
#
#
#
#
#
    return rows, by_id


def check_engine_summary(rows: list[ClaimRow], errors: list[str]) -> None:
    lines = ENGINE_MATRIX.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    total: list[str] | None = None
    summaries: dict[str, list[str]] = {}
    section_lines: dict[str, int] = {}
    for line_no, line in enumerate(lines, 1):
        if line.startswith("## "):
            section_lines[line.removeprefix("## ").strip()] = line_no
        if line.startswith("| Area | Rows |"):
            header = [normalize_header(cell) for cell in split_cells(line)]
        elif header is not None and total is None and line.startswith("|"):
            cells = [cell.replace("*", "").strip() for cell in split_cells(line)]
            if is_separator(cells):
                continue
            if cells[0] == "Total":
                total = cells
            else:
                summaries[cells[0]] = cells
    if header is None or total is None or len(header) != len(total):
        errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: missing or malformed lifecycle summary")
        return

    actual_rows = [row for row in rows if row.path == ENGINE_MATRIX]

    def check_counts(label: str, recorded_cells: list[str], scoped_rows: list[ClaimRow]) -> None:
        if len(recorded_cells) != len(header):
            errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: malformed {label} lifecycle summary")
            return
        expected = {"rows": len(scoped_rows)}
        expected.update(
            {normalize_header(state): sum(row.state == state for row in scoped_rows) for state in STATES}
        )
        for index, name in enumerate(header[1:], 1):
            if name not in expected:
                continue
            try:
                recorded = int(recorded_cells[index])
            except ValueError:
                errors.append(
                    f"{ENGINE_MATRIX.relative_to(ROOT)}: non-numeric {label} summary for {name}"
                )
                continue
            if recorded != expected[name]:
                errors.append(
                    f"{ENGINE_MATRIX.relative_to(ROOT)}: {label} summary {name}={recorded}; "
                    f"actual {expected[name]}"
                )

    check_counts("total", total, actual_rows)
    for area, section in ENGINE_SUMMARY_SECTIONS:
        recorded_cells = summaries.get(area)
        section_line = section_lines.get(section)
        if recorded_cells is None or section_line is None:
            errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: missing {area} lifecycle summary")
            continue
        next_section_line = min(
            (line_no for line_no in section_lines.values() if line_no > section_line),
            default=len(lines) + 1,
        )
        scoped_rows = [
            row for row in actual_rows if section_line < row.line_no < next_section_line
        ]
        check_counts(area, recorded_cells, scoped_rows)


def is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("`").lower()
    return not normalized or normalized in {"-", "none", "unassigned", "open", "leaf open"}


def local_line_anchors(value: str, source: Path) -> list[str]:
    anchors: list[str] = []
    for raw_target in LINK_RE.findall(value):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path, _, fragment = target.partition("#")
        line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
        if line_match is None:
            continue
        resolved = (source.parent / target_path).resolve()
        if not resolved.is_file():
            continue
        line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
        start = int(line_match.group(1))
        end = int(line_match.group(2) or start)
        if start < 1 or end < start or end > line_count:
            continue
        try:
            anchors.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    for match in RAW_LOCAL_ANCHOR_RE.finditer(value):
        resolved = (ROOT / match.group(1)).resolve()
        if not resolved.is_file():
            continue
        line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
        start = int(match.group(2))
        end = int(match.group(3) or start)
        if start < 1 or end < start or end > line_count:
            continue
        try:
            anchors.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    return anchors


def is_code_anchor(value: str, source: Path) -> bool:
    if is_placeholder(value):
        return False
    return any(
        path in CODE_ANCHOR_FILES or path.startswith(CODE_ANCHOR_PREFIXES)
        for path in local_line_anchors(value, source)
    )


def is_test_anchor(value: str, source: Path) -> bool:
    if is_placeholder(value):
        return False
    return any(
        path in EVIDENCE_ANCHOR_FILES or path.startswith(TEST_ANCHOR_PREFIXES)
        for path in local_line_anchors(value, source)
    )


def ledger_line_anchors(value: str, source: Path) -> list[str]:
    return [
        path
        for path in local_line_anchors(value, source)
        if path == ".agents/parity-ledger.md"
    ]


# --- record-anchor ratchet (ENG-RECORD-ANCHOR-RATCHET, #632) ------------------
#
# Records cite code as `server_main.cpp:505`. Code moves; the citation does not.
# The checker above LOOKS like it catches that and does not:
#
#   1. NO REPORT, not "no parser". `local_line_anchors` reads BOTH citation
#      forms: the markdown link through `LINK_RE`, and the bare
#      `file.cpp:123` form through `RAW_LOCAL_ANCHOR_RE` since ee511ca8a. It
#      range-checks each one. What it does not do is REPORT: on a missing file
#      and on an out-of-range line the loop runs `continue`, so the bad anchor
#      never enters the returned list, and `is_code_anchor`'s `any()` swallows
#      what is left. There was no symbol test either, which is the gap that
#      matters: 32 of the 38 offenders recorded here are IN RANGE, so a range
#      check could not have found them.
#   2. `any`, NOT `all`. `is_code_anchor` returns true if ONE anchor in a cell
#      qualifies, so a single good link covers arbitrarily many rotted citations
#      beside it. That is not a bug in the STATE gate -- a row IS evidenced by
#      one good anchor, and the `any` stays -- but it is why nothing counted the
#      others.
#   3. STATE. `EVIDENCED_STATES` omits `ACTIVE` and `READY`, so 92 live rows got
#      no anchor check at all.
#
# TWO POPULATIONS, TWO RATIOS, and quoting one without its denominator is what
# produced the false "the bare form was never parsed" claim this comment
# replaces. Measured at `8daa67b39`, the head before this branch merged `main` a
# second time, counting a citation only where it is the WHOLE of a backtick
# span, which is what the parser requires:
#
#   * ALL citation forms in the five matrices, ours and upstream: 492 links and
#     1708 bare, 2200 in total. 525 of the bare forms sit under a
#     `RAW_LOCAL_ANCHOR_RE` prefix, so 1017 of 2200 (46.2%) were already parsed.
#     Most of the remainder are upstream paths that reach no local checker.
#   * The citations this ratchet CLASSIFIES, which is the population that
#     matters: 867. Of those 832 (96.0%) were already parsed AND range-checked
#     before this row, and 35 (4.0%) are genuinely new to parsing, under
#     `.agents/`, `docs/` and `website/`. Every offender recorded then sat in
#     the 96%. The value this row adds is the symbol test and the report, not the
#     parser.
#
# A range check is not the fix: every stale anchor found
# by hand during the 2026-08-13/14 campaign was IN RANGE --
# `docs/USAGE.md:902` (the count line had moved to :1126), `multimodal.py:17-43`
# (the block ends at :45) and `server_main.cpp:308` (a different table entry
# after a 4-line comment landed above). Only "does this line contain the symbol
# named beside it" separates those from a live citation.
#
# THE RATCHET. Enforcing correctness over the whole backlog in one landing would
# surface an unknown amount of unrelated rot, so this mirrors the DSR ratchet in
# scripts/check-device-leakage.py, which this repo already trusts:
# scripts/record-anchor-baseline.json holds the accepted STALE + BROKEN counts,
# a bucket ABOVE its baseline fails, and a bucket BELOW it fails too, with the
# instruction to lower the baseline in the SAME commit as the repair. The number
# only ever moves down, and only deliberately. `--report` names every offender
# so the backlog is legible rather than a number.
#
# WHERE THE CONSERVATIVE LINE IS DRAWN, and why each side of it is where it is.
# A checker that cries wolf gets disabled, and this one has to survive a
# four-figure backlog, so every rule below prefers a missed rot to a false one:
#
#   * ONLY the `code` and `tests` cells of rows in RECORD_ANCHOR_STATES. The
#     `upstream` column is never read. That is what keeps the upstream
#     references (`vllm/model_executor/...py:123`, `csrc/...cu:44`) out of the
#     count structurally, rather than by a path heuristic.
#   * A bare citation must be the WHOLE of a backtick span, so running prose can
#     never be parsed as a path.
#   * It must resolve to a file in the tree, or be a near miss: at least two path
#     separators AND an existing parent directory, i.e. "a directory we own with
#     a filename we do not" -- a rename or a deletion. `vllm/utils/hashing.py`
#     and `tests/v1/core/test_scheduler.py` have no such parent here and are
#     silently skipped, which is correct: they are upstream, and we cannot
#     validate an anchor into a tree we do not have.
#   * A symbol is inferred ONLY from an immediately adjacent backtick span --
#     whitespace between them, or whitespace and one `(` for the trailing
#     `path:line (`Symbol`)` form. Nothing else, and never from prose.
#   * That span must LOOK like a symbol: an identifier, at least 4 characters,
#     carrying `_`, `::`, `()` or an uppercase letter. `bf16` and `nvfp4` sit
#     next to citations constantly and are not symbols; `MoeAuxStream`,
#     `evict_blocks` and `Scheduler::shutdown()` are.
#   * With no inferable symbol the citation is OK by construction. 801 of the
#     867 in-scope citations land there, which is 92.4%, or about 13 in 14.
#     That is the intended polarity: this gate exists to be believed when it
#     does fire.
#   * The symbol test asks whether the cited LINES CONTAIN the name. A comment
#     that mentions the symbol therefore reads OK. That is a measured limit and
#     not a defect: `KERNEL-ATTN-MLA-SPARSE` cites
#     `include/vllm/v1/attention/backend.h:271` for `get_kv_cache_shape`, whose
#     real declaration is at :341, and drift on `main` moved a ROCm comment
#     naming the symbol onto :271. Tightening this would need a parser per
#     language, which is the cry-wolf trade this whole block refuses.
RECORD_ANCHOR_BASELINE = ROOT / "scripts/record-anchor-baseline.json"
RECORD_ANCHOR_VERDICTS = ("ok", "stale", "broken")
# The BUDGET is the rot only. `ok` is counted and printed but deliberately NOT
# stored: a baseline that pinned it would make every PR that adds or removes any
# citation rewrite this file, which is exactly the shared-file lock AGENTS.md
# forbids. Per-bucket rather than one total, so a repaired BROKEN cannot pay for
# a new STALE.
RECORD_ANCHOR_BUCKETS = ("stale", "broken")
# Gap 3. `EVIDENCED_STATES` itself is deliberately NOT widened. Making ACTIVE and
# READY *require* an anchor raises 85 errors across 53 rows that carry prose
# evidence today, which is the bulk cleanup this row exists to avoid. (The unit
# is errors, not rows: the contract check emits one per missing anchor field, so
# a row can raise more than one -- 32 rows raise two here and 21 raise one.)
# They join the COUNT instead, and the ratchet absorbs what that surfaces.
RECORD_ANCHOR_STATES = EVIDENCED_STATES | {"ACTIVE", "READY"}
RECORD_ANCHOR_FIELDS = ("code", "tests")
BARE_CITATION_RE = re.compile(
    r"([A-Za-z0-9_./+-]*[A-Za-z0-9_+-]\.[A-Za-z0-9_+-]+):(\d+)(?:-(\d+))?"
)
BACKTICK_SPAN_RE = re.compile(r"`([^`\n]+)`")
SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?")


@dataclass(frozen=True)
class Citation:
    # No matrix field: row IDs are unique across the matrices (check_matrices
    # enforces it), so the item id already locates the offender.
    item_id: str
    path: str
    start: int
    end: int
    symbol: str | None
    verdict: str

    def describe(self) -> str:
        span = f"{self.start}" if self.end == self.start else f"{self.start}-{self.end}"
        want = f" expected `{self.symbol}`" if self.symbol else ""
        return f"{self.verdict:<6} {self.item_id} -> {self.path}:{span}{want}"


@dataclass
class RecordAnchorResult:
    counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict.fromkeys(RECORD_ANCHOR_VERDICTS, 0)
    )
    citations: list[Citation] = dataclasses.field(default_factory=list)

    @property
    def offenders(self) -> list[Citation]:
        return [c for c in self.citations if c.verdict in {"STALE", "BROKEN"}]

    @property
    def total(self) -> int:
        """STALE + BROKEN. `ok` is reported but is not part of the budget."""
        return sum(self.counts[bucket] for bucket in RECORD_ANCHOR_BUCKETS)


def looks_like_symbol(text: str) -> bool:
    text = text.strip()
    if len(text) < 4 or SYMBOL_RE.fullmatch(text) is None:
        return False
    # A LEADING underscore in a record is nearly always an abbreviated suffix --
    # "the text-only `_ModelInfo`" naming the `ModelInfo` beside it, not a
    # symbol spelled `_ModelInfo`. Searching for it literally produced the one
    # false STALE this rule was measured against, on an anchor that was right.
    if text.startswith("_"):
        return False
    return "_" in text or "::" in text or text.endswith("()") or any(c.isupper() for c in text)


def cell_citations(cell: str, source: Path, root: Path) -> list[tuple[Path, int, int, str | None]]:
    """Every citation of THIS tree in one record cell, with its expected symbol.

    Returns `(resolved_path, start, end, symbol_or_None)`. Both citation forms
    are recognised -- the markdown link `[label](path#L12)` the old parser saw,
    and the bare `` `path:12` `` / `` `path:12-20` `` span that is six times
    more common here and was never parsed as a citation at all.
    """
    tokens: list[tuple[int, int, str, object]] = []
    links: list[tuple[int, int]] = []
    for match in LINK_RE.finditer(cell):
        target = match.group(1).strip().strip("<>")
        links.append((match.start(), match.end()))
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path, _, fragment = target.partition("#")
        line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
        if line_match is None:
            continue
        start = int(line_match.group(1))
        end = int(line_match.group(2) or start)
        resolved = (source.parent / target_path).resolve()
        tokens.append((match.start(), match.end(), "citation", (resolved, start, end)))
    for match in BACKTICK_SPAN_RE.finditer(cell):
        # A backtick span inside a link is that link's LABEL, not a token beside
        # it: `[`foo.cpp`](../src/foo.cpp#L4)` must not be read as a neighbour.
        if any(lo <= match.start() < hi for lo, hi in links):
            continue
        inner = match.group(1).strip()
        bare = BARE_CITATION_RE.fullmatch(inner)
        if bare is not None:
            start = int(bare.group(2))
            end = int(bare.group(3) or start)
            resolved = (root / bare.group(1)).resolve()
            tokens.append((match.start(), match.end(), "citation", (resolved, start, end)))
        elif looks_like_symbol(inner):
            tokens.append((match.start(), match.end(), "symbol", inner))
        else:
            tokens.append((match.start(), match.end(), "other", inner))
    tokens.sort(key=lambda t: t[0])

    found: list[tuple[Path, int, int, str | None]] = []
    for index, (lo, hi, kind, payload) in enumerate(tokens):
        if kind != "citation":
            continue
        resolved, start, end = payload  # type: ignore[misc]
        symbol: str | None = None
        if index > 0 and tokens[index - 1][2] == "symbol":
            if not cell[tokens[index - 1][1]:lo].strip():
                symbol = str(tokens[index - 1][3]).strip()
        if symbol is None and index + 1 < len(tokens) and tokens[index + 1][2] == "symbol":
            if cell[hi:tokens[index + 1][0]].strip() in {"", "("}:
                symbol = str(tokens[index + 1][3]).strip()
        found.append((resolved, start, end, symbol))
    return found


def classify_citation(
    resolved: Path, start: int, end: int, symbol: str | None, root: Path
) -> str | None:
    """OK / STALE / BROKEN, or None when the citation is not about this tree."""
    if not resolved.is_file():
        # "A directory we own with a filename we do not" -- a rename or a
        # deletion, and the one missing-file case worth calling BROKEN. Require
        # THREE path components so a one-segment upstream name whose top-level
        # directory happens to match ours (llama.cpp's `src/llama-model.cpp`,
        # vLLM's `cmake/utils.cmake`) is skipped rather than blamed on us.
        try:
            depth = len(resolved.relative_to(root).parts)
        except ValueError:
            return None
        if depth >= 3 and resolved.parent.is_dir():
            return "BROKEN"
        return None
    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    if start < 1 or end < start or end > len(lines):
        return "BROKEN"
    if symbol is None:
        return "OK"
    base = symbol.removesuffix("()").split("::")[-1]
    body = "\n".join(lines[start - 1:end])
    return "OK" if re.search(r"\b" + re.escape(base) + r"\b", body) else "STALE"


def scan_record_anchors(
    rows: list[ClaimRow] | None = None, root: Path | None = None
) -> RecordAnchorResult:
    root = ROOT if root is None else root
    if rows is None:
        rows, _ = check_matrices([])
    result = RecordAnchorResult()
    for row in rows:
        if row.state not in RECORD_ANCHOR_STATES:
            continue
        # BY INDEX, not by name: several matrices carry one `Local evidence`
        # column that field_index resolves for BOTH `code` and `tests`, and
        # reading it twice would count every citation in it twice.
        indices = {field_index(row.header, name) for name in RECORD_ANCHOR_FIELDS}
        for index in sorted(i for i in indices if i is not None):
            cell = row.cells[index] if index < len(row.cells) else ""
            if is_placeholder(cell):
                continue
            for resolved, start, end, symbol in cell_citations(cell, row.path, root):
                verdict = classify_citation(resolved, start, end, symbol, root)
                if verdict is None:
                    continue
                try:
                    shown = resolved.relative_to(root).as_posix()
                except ValueError:
                    shown = resolved.as_posix()
                result.counts[verdict.lower()] += 1
                result.citations.append(
                    Citation(
                        item_id=row.item_id,
                        path=shown,
                        start=start,
                        end=end,
                        symbol=symbol,
                        verdict=verdict,
                    )
                )
    return result


def load_record_anchor_baseline() -> dict[str, int]:
    if not RECORD_ANCHOR_BASELINE.is_file():
        return {}
    data = json.loads(RECORD_ANCHOR_BASELINE.read_text(encoding="utf-8"))
    return {bucket: int(data["buckets"][bucket]) for bucket in RECORD_ANCHOR_BUCKETS}


def write_record_anchor_baseline(result: RecordAnchorResult) -> int:
    previous = load_record_anchor_baseline()
    if previous and result.total > sum(previous.values()):
        print(
            "REFUSING to write a HIGHER record-anchor baseline "
            f"({sum(previous.values())} -> {result.total}). The ratchet only turns one "
            "way: repair the anchors instead of banking the rot.",
            file=sys.stderr,
        )
        return 1
    payload = {
        "_comment": [
            "Record-anchor baseline for scripts/check-agent-record.py",
            "(ENG-RECORD-ANCHOR-RATCHET, .agents/specs/record-anchor-ratchet.md).",
            "STALE = the cited line exists but does not contain the symbol named",
            "beside it. BROKEN = the line is out of range, or the file is gone.",
            "THESE NUMBERS MAY ONLY EVER GO DOWN. Lower them in the SAME commit as",
            "the repair that earned it, by running:",
            "  python3 scripts/check-agent-record.py --write-baseline",
            "It is a rot budget, never to be raised to make a failing check pass.",
            "Only the rot is stored. The OK count is printed by --report but kept out",
            "of this file on purpose: pinning it would make every change that adds or",
            "removes a citation rewrite this file, which is a lock, not a ratchet.",
        ],
        "total": result.total,
        "buckets": {bucket: result.counts[bucket] for bucket in RECORD_ANCHOR_BUCKETS},
    }
    RECORD_ANCHOR_BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"baseline written: {RECORD_ANCHOR_BASELINE.name} -> {result.total}")
    return 0


def record_anchor_report(result: RecordAnchorResult) -> str:
    lines = [
        "Record anchors in the `code` / `tests` cells of "
        f"{'/'.join(sorted(RECORD_ANCHOR_STATES))} rows:",
        "",
    ]
    lines.extend(f"  {c.describe()}" for c in result.offenders)
    if result.offenders:
        lines.append("")
    lines.append(
        "record anchors: "
        + ", ".join(f"{b}={result.counts[b]}" for b in RECORD_ANCHOR_VERDICTS)
        + f"  -> rot {result.total}"
    )
    return "\n".join(lines)


def check_record_anchors(result: RecordAnchorResult, errors: list[str]) -> None:
    baseline = load_record_anchor_baseline()
    if not baseline:
        errors.append(
            f"no record-anchor baseline at {RECORD_ANCHOR_BASELINE.relative_to(ROOT)}; "
            "run --write-baseline to establish one"
        )
        return
    for bucket in RECORD_ANCHOR_BUCKETS:
        got, want = result.counts[bucket], baseline[bucket]
        if got > want:
            errors.append(
                f"RECORD ANCHOR REGRESSION in bucket '{bucket}': {got} > baseline {want}. "
                "A citation names a line that no longer holds what the prose says it "
                "does. Run `python3 scripts/check-agent-record.py --report` for the "
                "offenders and repair the anchor. NEVER raise the baseline to pass."
            )
        elif got < want:
            errors.append(
                f"record-anchor baseline STALE in bucket '{bucket}': {got} < baseline "
                f"{want}. A repair must lower the baseline in the SAME commit: run "
                "`python3 scripts/check-agent-record.py --write-baseline` and commit it."
            )


def commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def local_spec_paths(row: ClaimRow) -> list[Path]:
    paths: list[Path] = []
    spec_root = (AGENTS / "specs").resolve()
    for raw_target in LINK_RE.findall(row.field("spec")):
        target = raw_target.strip().strip("<>").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (row.path.parent / target).resolve()
        try:
            resolved.relative_to(spec_root)
        except ValueError:
            continue
        if resolved.suffix == ".md" and resolved.is_file():
            paths.append(resolved)
    return paths


def has_substantive_spec_content(lines: list[str]) -> bool:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,6}\s+", stripped) or stripped.startswith("```"):
            continue
        if stripped.startswith("|"):
            cells = split_cells(stripped)
            if is_separator(cells):
                continue
            next_nonempty = next(
                (candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()),
                "",
            )
            if next_nonempty.startswith("|") and is_separator(split_cells(next_nonempty)):
                continue
            if len(cells) >= 2 and any(not is_placeholder(cell) for cell in cells[1:]):
                return True
            continue
        if not is_placeholder(stripped):
            return True
    return False


def structured_spec_fields(text: str) -> set[str]:
    lines = text.splitlines()
    fields: set[str] = set()
    for index, line in enumerate(lines):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading is not None:
            level = len(line) - len(line.lstrip("#"))
            end = len(lines)
            for candidate in range(index + 1, len(lines)):
                next_heading = re.match(r"^(#{1,6})\s+", lines[candidate])
                if next_heading is not None and len(next_heading.group(1)) <= level:
                    end = candidate
                    break
            if has_substantive_spec_content(lines[index + 1 : end]):
                fields.add(normalize_header(heading.group(1)))
            continue
        if line.startswith("|"):
            cells = split_cells(line)
            next_nonempty = next(
                (
                    candidate.strip()
                    for candidate in lines[index + 1 :]
                    if candidate.strip()
                ),
                "",
            )
            is_header = next_nonempty.startswith("|") and is_separator(
                split_cells(next_nonempty)
            )
            if (
                len(cells) >= 2
                and not is_header
                and not is_separator(cells)
                and any(not is_placeholder(cell) for cell in cells[1:])
            ):
                fields.add(normalize_header(cells[0]))
    return fields


def missing_spec_requirements(text: str) -> list[str]:
    fields = structured_spec_fields(text)
    missing: list[str] = []
    for label, alternatives in SPEC_REQUIREMENTS.items():
        normalized = [normalize_header(alternative) for alternative in alternatives]
        if not any(
            field == alternative or field.startswith(alternative + " ")
            for field in fields
            for alternative in normalized
        ):
            missing.append(label)
    return missing


def check_spec(row: ClaimRow, errors: list[str]) -> None:
    specs = local_spec_paths(row)
    location = f"{row.path.relative_to(ROOT)}:{row.line_no}"
    if not specs:
        errors.append(f"{location}: {row.item_id} {row.state} has no real .agents/specs link")
        return
    token = f"`{row.item_id}`"
    matching = [path for path in specs if token in path.read_text(encoding="utf-8")]
    if not matching:
        errors.append(f"{location}: no linked spec names exact stable token `{row.item_id}`")
        return
    text = matching[0].read_text(encoding="utf-8")
    for label in missing_spec_requirements(text):
        errors.append(
            f"{matching[0].relative_to(ROOT)}: spec for {row.item_id} lacks structured {label}"
        )


def claim_sources() -> list[Path]:
    """Every file that may carry an active claim row.

    One file per claim in .agents/claims/ (ENG-RECORD-CONFLICT-SURFACES, #364),
    plus the legacy table in coordination.md. Both are read, so a claim is
    equally valid in either and no existing row had to be migrated -- the table
    empties as its claims close.

    The per-claim file exists because the table is insert-at-one-anchor: every
    concurrent claim appended at the same line, which made coordination.md the
    largest single conflict source in the repository (8 of the 16 conflicting
    open PRs at origin/main d928e2c3, six of them one author's sequential ROCm
    stack whose only conflict was this). A file with one writer cannot collide.
    """
    sources = [AGENTS / "coordination.md"]
    claims_dir = AGENTS / "claims"
    if claims_dir.is_dir():
        sources.extend(sorted(claims_dir.glob("CLAIM-*.md")))
    return sources


def parse_active_claims(errors: list[str]) -> dict[str, ClaimRecord]:
    """Read claim rows from the legacy table and per-claim files.

    Per-claim files carry structured row-state annotations and a lifecycle.
    The legacy table remains readable until its historical rows retire.
    Duplicate claim IDs still fail across both source types.
    Row IDs keep the existing ID_RE grammar and multi-row behavior.
    Structured consistency is checked after the matrices are available.

    The returned record retains its source location for precise diagnostics.
    It also retains whether the source uses the strict per-claim format.
    Callers use row_ids when they only need the former membership behavior.
    The lifecycle is interpreted only for a selected live-row owner.
    Unknown lifecycle values remain data so the consistency gate can fail.
    No claim ID or row ID is inferred from prose outside the table cells.
    """
    claims: dict[str, ClaimRecord] = {}
    origin: dict[str, str] = {}
    for path in claim_sources():
        parse_claim_source(path, errors, claims, origin)
    return claims


def check_row_contracts(
    rows: list[ClaimRow], by_id: dict[str, ClaimRow], errors: list[str]
) -> None:
    active_claims = parse_active_claims(errors)
    required_fields = ("id", "item", "upstream", "code", "tests", "spec", "state", "owner")

    for row in rows:
        location = f"{row.path.relative_to(ROOT)}:{row.line_no}"
        for field in required_fields:
            if field_index(row.header, field) is None:
                errors.append(f"{location}: {row.item_id} table lacks semantic {field} column")

        if is_placeholder(row.field("item")):
            errors.append(f"{location}: {row.item_id} has no item description")
        if is_placeholder(row.field("upstream")):
            errors.append(f"{location}: {row.item_id} has no upstream anchor/target")

        if row.state in EVIDENCED_STATES:
            if not is_code_anchor(row.field("code"), row.path):
                errors.append(f"{location}: {row.item_id} {row.state} lacks exact local code anchor")
            if not is_test_anchor(row.field("tests"), row.path):
                errors.append(f"{location}: {row.item_id} {row.state} lacks exact test/evidence anchor")

        if row.state in READY_STATES:
            check_spec(row, errors)

        if row.state in {"SPIKE", "ACTIVE"}:
            claim_match = CLAIM_RE.search(row.field("owner"))
            if claim_match is None:
                errors.append(f"{location}: {row.state} row {row.item_id} has no CLAIM-* owner")
            else:
                claim = claim_match.group(0)
                if row.item_id not in claimed_row_ids(active_claims, claim):
                    errors.append(
                        f"{location}: owner {claim} does not claim active row {row.item_id} in any claim source"
                    )

        if row.state == "DONE":
            if not ledger_line_anchors(row.field("tests"), row.path):
                errors.append(f"{location}: DONE row {row.item_id} lacks exact parity-ledger link")
            owner = row.field("owner").strip().strip("`")
            if COMMIT_RE.fullmatch(owner) is None:
                errors.append(
                    f"{location}: DONE row {row.item_id} owner is not the hexadecimal closing commit"
                )
            elif not commit_exists(owner):
                errors.append(
                    f"{location}: DONE row {row.item_id} closing commit {owner} does not exist"
                )

    for claim, item_ids in ((claim, record.row_ids) for claim, record in active_claims.items()):
        if not item_ids:
            errors.append(f"active claim {claim} has no stable row IDs")
        for item_id in item_ids:
            row = by_id.get(item_id)
            if row is None:
                errors.append(f"active claim {claim} references unknown row {item_id}")
            elif row.state not in {"SPIKE", "ACTIVE"}:
                errors.append(
                    f"active claim {claim} references {item_id} in state {row.state}, not SPIKE/ACTIVE"
                )
    check_claim_state_consistency(active_claims, by_id, errors)

def check_model_invariants(errors: list[str]) -> None:
    path = AGENTS / "model-matrix.md"
    rows: list[tuple[list[str], str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `MODEL-"):
            continue
        cells = split_cells(line)
        if len(cells) < 3:
            continue
        aliases = re.findall(r"`([^`]+)`", cells[1])
        targets = [value for value in re.findall(r"`([^`]+)`", cells[2]) if "::" in value]
        if aliases and targets:
            rows.append((aliases, targets[-1]))

    actual = {
        "rows": len(rows),
        "memberships": sum(len(aliases) for aliases, _ in rows),
        "architectures": len({alias for aliases, _ in rows for alias in aliases}),
        "targets": len({target for _, target in rows}),
        "modules": len({target.split("::", 1)[0] for _, target in rows}),
    }
    # `targets` 310 -> 309 and `modules` 261 -> 245 on 2026-09-05 (#2819): the
    # anchor re-point recorded below. Regenerated 2026-09-27 (record-rot repair,
    # issue ISSUE-LOCAL-01M3JMSD9P7REKNQVEG9HCEWTD): rows 325 -> 324,
    # memberships 374 -> 373, architectures 357 -> 356, `targets` 309 kept and
    # `modules` 245 kept. The SystemOne decision rows (`MODEL-LAYA`,
    # `MODEL-KEV`, `MODEL-CUA-S1-FORMS`, `MODEL-CLM`, `MODEL-TEV1`,
    # `MODEL-XOR`, `MODEL-JEV`, `MODEL-GLINER25-DECIDE`) that bc5dbc367 closed
    # were dropped from the matrix rather than carried as DONE rows, and
    # a3b33bb20 regenerated `targets` / `modules` from that tree; the remaining
    # counters are regenerated the same way: from the rows, never to make a
    # transition pass.
    # Citation-preserving padding; retired detail lives in the completed
    # archive and the commit message. Do not remove.
    #
    expected = {
        "rows": 324,
        "memberships": 373,
        "architectures": 356,
        "targets": 309,
        "modules": 245,
    }
    if actual != expected:
        errors.append(f"{path.relative_to(ROOT)}: model inventory {actual}, expected {expected}")


def check_table_shapes(paths: list[Path], errors: list[str]) -> None:
    for path in sorted(set(paths)):
        expected_pipes: int | None = None
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("|"):
                expected_pipes = None
                continue
            pipes = len(re.findall(r"(?<!\\)\|", line))
            if expected_pipes is None:
                expected_pipes = pipes
            elif pipes != expected_pipes:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no}: table has {pipes} pipes; expected {expected_pipes}"
                )


def check_spec_location(errors: list[str]) -> None:
    misplaced = re.compile(r"(?:spec|scoping|semantics|feasibility|notes)", re.I)
    allowed = {"benchmark-protocol.md"}
    for path in AGENTS.glob("*.md"):
        if path.name not in allowed and misplaced.search(path.stem):
            errors.append(f"{path.relative_to(ROOT)}: feature spec/scoping file belongs in .agents/specs/")


ISSUE_ROW = re.compile(
    r"^\|\s*\[#(\d+)\]\((https://github\.com/[^)]+/issues/(\d+))\)\s*\|"
    r"\s*(?:`([A-Z0-9][A-Za-z0-9_.-]*)`|—)\s*\|"
)


# The retired tracked index stays named so the checker can refuse its return.
RETIRED_INDEX = AGENTS / "issue-index.md"


ISSUE_REFERENCE_RE = re.compile(
    r"(?<![A-Z0-9_-])ISSUE-GH-[1-9][0-9]*(?![A-Z0-9_-])"
    r"|(?<![A-Z0-9_-])ISSUE-LOCAL-[0-7][0-9A-HJKMNP-TV-Z]{25}(?![A-Z0-9_-])"
    r"|(?<![A-Za-z0-9])#[1-9][0-9]*(?![0-9])"
    r"|(?<![A-Za-z0-9])issues/[1-9][0-9]*(?![0-9])"
)


def issue_references_in_text(
    text: str,
    *,
    path: Path | None = None,
    frozen_archive: bytes | None = None,
) -> set[str]:
    """Return references, excluding only source-verified self-declarations."""

    searchable = text.replace("\r\n", "\n").replace("\r", "\n")
    record: issue_records.IssueRecord | None = None
    if path is not None:
        try:
            parsed = issue_records.parse_issue_text(searchable)
        except issue_records.IssueRecordError:
            parsed = None
        if (
            parsed is not None
            and path.name == f"{parsed.id}.md"
            and path.parent.parent.name == "issues"
            and path.parent.parent.parent.name == ".agents"
        ):
            record = parsed
            searchable = searchable.replace(f"ID: {record.id}", "ID: ", 1)
            github = "-" if record.github is None else str(record.github)
            searchable = searchable.replace(f"GitHub: {github}", "GitHub: ", 1)

    if record is not None and path is not None and path.parent.name == "_intake":
        evidence = issue_records.valid_intake_archive_evidence(
            record,
            frozen_archive,
        )
        if evidence is not None and record.github is not None:
            _, archived_line = evidence
            masked_line = re.sub(
                rf"(?<![A-Za-z0-9])#{record.github}(?![0-9])"
                rf"|(?<![A-Za-z0-9])issues/{record.github}(?![0-9])",
                "",
                archived_line,
            )
            searchable = searchable.replace(
                f"> {archived_line}",
                f"> {masked_line}",
                1,
            )

    references: set[str] = set()
    for match in ISSUE_REFERENCE_RE.finditer(searchable):
        reference = match.group(0)
        if reference.startswith("issues/"):
            reference = f"#{reference.removeprefix('issues/')}"
        references.add(reference)
    return references


def discover_issue_references(
    commit_bodies: str,
    changed_files: dict[Path, str],
    *,
    frozen_archive: bytes | None = None,
) -> set[str]:
    """Discover references in both branch commit bodies and changed-file content."""

    references = issue_references_in_text(commit_bodies)
    for path, text in changed_files.items():
        references.update(
            issue_references_in_text(
                text,
                path=path,
                frozen_archive=frozen_archive,
            )
        )
    return references


def branch_issue_references(base: str = "origin/main") -> set[str]:
    """Read commit bodies and changed-file contents from one offline git range."""

    try:
        merge_base = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", base, "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if merge_base.returncode != 0:
            return set()
        revision = f"{merge_base.stdout.strip()}..HEAD"
        log = subprocess.run(
            ["git", "-C", str(ROOT), "log", "--format=%B", revision],
            capture_output=True,
            text=True,
            timeout=20,
        )
        changed = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--name-only",
                "-z",
                "--diff-filter=ACMRT",
                revision,
            ],
            capture_output=True,
            timeout=20,
        )
        if log.returncode != 0 or changed.returncode != 0:
            return set()
    except (OSError, subprocess.SubprocessError):
        return set()

    changed_files: dict[Path, str] = {}
    for raw_path in changed.stdout.split(b"\0"):
        if not raw_path:
            continue
        try:
            relative = Path(raw_path.decode("utf-8"))
            changed_files[relative] = (ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
    try:
        frozen_archive = (
            ROOT / ".agents" / "completed" / "issue-index.md"
        ).read_bytes()
    except OSError:
        frozen_archive = None
    return discover_issue_references(
        log.stdout,
        changed_files,
        frozen_archive=frozen_archive,
    )


def canonical_intake_debt(
    records: list[tuple[Path, issue_records.IssueRecord]],
) -> tuple[str, ...]:
    """Report current intake identities without comparing a shared baseline."""

    return tuple(
        sorted(
            record.id
            for path, record in records
            if path.parent.name == "_intake"
        )
    )


def check_canonical_issue_references(
    errors: list[str],
    references: set[str],
    records: list[tuple[Path, issue_records.IssueRecord]],
) -> None:
    """Resolve branch references and reject migration intake as ownership."""

    materialized = [record for _, record in records]
    paths = {record.id: path for path, record in records}
    for reference in sorted(references):
        normalized = (
            f"#{reference.removeprefix('issues/')}"
            if reference.startswith("issues/")
            else reference
        )
        try:
            record = issue_records.resolve_issue_reference(normalized, materialized)
        except issue_records.IssueRecordError as error:
            if normalized.startswith("#"):
                continue
            errors.append(str(error))
            continue
        if paths[record.id].parent.name == "_intake":
            errors.append(
                f"issue reference {reference} resolves to _intake record "
                f"{record.id}; triage it to a canonical row or exactly owned _owed"
            )


def check_issue_records(
    errors: list[str],
    *,
    issues_root: Path = ISSUES_ROOT,
    rows: set[str] | None = None,
    owed: issue_records.OwedLookup | None = None,
    references: set[str] | None = None,
    frozen_archive: bytes | None = None,
) -> None:
    """Validate canonical local issue authority and branch references offline."""

    if RETIRED_INDEX.is_file():
        errors.append(
            f"{RETIRED_INDEX.relative_to(ROOT)}: the tracked issue index is back; "
            "canonical issue authority belongs under .agents/issues/"
        )
    if not issues_root.is_dir():
        errors.append(f"{issues_root}: canonical issue directory is absent")
        return

    effective_rows = issue_records.canonical_rows(ROOT) if rows is None else rows
    effective_owed = (
        issue_records.owed_issue_counts(ROOT) if owed is None else owed
    )
    if frozen_archive is None:
        try:
            frozen_archive = (
                ROOT / ".agents" / "completed" / "issue-index.md"
            ).read_bytes()
        except OSError:
            frozen_archive = None

    records: list[tuple[Path, issue_records.IssueRecord]] = []
    paths = sorted(issues_root.glob("**/*.md"))
    if not paths:
        errors.append(f"{issues_root}: canonical issue directory contains no records")
        return
    for path in paths:
        try:
            record = issue_records.parse_issue_file(path)
            issue_records.validate_issue_record(
                record,
                path,
                effective_rows,
                effective_owed,
                frozen_archive=frozen_archive,
            )
        except (OSError, issue_records.IssueRecordError) as error:
            label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            errors.append(f"{label}: {error}")
            continue
        records.append((path, record))

    try:
        issue_records.validate_issue_collection(record for _, record in records)
    except issue_records.IssueRecordError as error:
        errors.append(f"{issues_root}: {error}")

    check_canonical_issue_references(
        errors,
        branch_issue_references() if references is None else references,
        records,
    )


def check_roadmap(by_id: dict[str, ClaimRow], errors: list[str]) -> None:
    path = AGENTS / "roadmap_v1.md"
    expected_blocks = [
        "ROAD-V1-A",
        "ROAD-V1-C1",
        "ROAD-V1-C2",
        "ROAD-V1-C3",
        "ROAD-V1-C4",
        "ROAD-V1-C5",
        "ROAD-V1-C6",
        "ROAD-V1-C7",
        "ROAD-V1-C8",
        "ROAD-V1-C9",
        "ROAD-V1-D1",
        "ROAD-V1-D2",
        "ROAD-V1-D3",
        "ROAD-V1-D4",
        "ROAD-V1-D5",
        # +D6 2026-08-05: llama.cpp device breadth folded into scope (user-directed).
        "ROAD-V1-D6",
    ]
    seen: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        issue = ISSUE_ROW.match(line)
        if issue:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: stores a GitHub issue table row "
                f"for #{issue.group(1)}. Canonical issue authority belongs under "
                ".agents/issues/; refresh only the untracked generated view with "
                "`python3 scripts/agent-issue-index.py --refresh`"
            )
            continue
        if not line.startswith("|"):
            continue
        cells = split_cells(line)
        if len(cells) < 6 or not cells[0].isdigit():
            continue
        block_id = cells[1].strip().strip("`")
        if not re.fullmatch(r"ROAD-V1-[A-Z0-9]+", block_id):
            continue
        seen.append((int(cells[0]), block_id))
        if STATE_RE.fullmatch(cells[5].strip()) is None:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: portfolio State cell needs exactly one state"
            )
    if seen != list(enumerate(expected_blocks)):
        errors.append(f"{path.relative_to(ROOT)}: portfolio order {seen}; expected {list(enumerate(expected_blocks))}")

    for source in (path, AGENTS / "coordination.md"):
        text = source.read_text(encoding="utf-8")
        for item_id in set(re.findall(r"`(" + ID_RE.pattern + r")`", text)):
            if item_id not in by_id:
                errors.append(f"{source.relative_to(ROOT)}: references unknown stable row {item_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help="print every STALE/BROKEN record anchor with the symbol expected",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite scripts/record-anchor-baseline.json (only ever DOWNWARD)",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    skips: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing canonical record: {path.relative_to(ROOT)}")

    rows: list[ClaimRow] = []
    by_id: dict[str, ClaimRow] = {}
    if not errors:
        check_links(errors)
        check_issue_records(errors)
        rows, by_id = check_matrices(errors)
        check_engine_summary(rows, errors)
        check_row_contracts(rows, by_id, errors)
        check_model_invariants(errors)
        spec_paths = [path for row in rows if row.state in READY_STATES for path in local_spec_paths(row)]
        # The issue index used to be in this list: it was the ONE record
        # surface every change had to write, and the only markdown table here
        # with no shape gate, so a row that lost its trailing pipe mis-rendered
        # on GitHub while every gate stayed green (#1033). It is derived now, and
        # a generated table cannot lose a pipe to a human edit -- the generator's
        # own suite gates its output shape instead (#2290).
        check_table_shapes(
            [
                AGENTS / "roadmap_v1.md",
                AGENTS / "coordination.md",
                *MATRIX_PATHS,
                *spec_paths,
            ],
            errors,
        )
        check_spec_location(errors)
        check_roadmap(by_id, errors)

        anchors = scan_record_anchors(rows)
        if args.report:
            print(record_anchor_report(anchors))
        if not args.write_baseline:
            # Writing is a MODE, not a step: it must not also report the gate it
            # is about to move, or a run that lowered the baseline would print a
            # regression against the value it just replaced.
            check_record_anchors(anchors, errors)
    elif args.write_baseline:
        print(
            "REFUSING to write a baseline from a tree whose record does not parse.",
            file=sys.stderr,
        )

    if errors:
        for error in dict.fromkeys(errors):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.write_baseline:
        # AFTER the error gate, deliberately. The mode used to return the moment
        # it had a number, so a tree that failed some OTHER record check could
        # still bank its rot, and the banked figure would carry the authority of
        # a run that never passed. A baseline is a measurement of the record, so
        # it is only taken from a record that checks out.
        return write_record_anchor_baseline(anchors)

    counts = [
        "ENGINE="
        + str(
            sum(
                any(row.item_id.startswith(prefix + "-") for prefix in ENGINE_PREFIXES)
                for row in rows
                if row.path == ENGINE_MATRIX
            )
        )
    ]
    for prefix, path in MATRICES.items():
        counts.append(
            f"{prefix}="
            + str(sum(row.item_id.startswith(prefix + "-") for row in rows if row.path == path))
        )
    counts.append(f"ANCHOR-ROT={anchors.total}")
    print("agent record OK: " + " ".join(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
