#!/usr/bin/env python3
"""Pure parsing and validation helpers for canonical local issue records.

This module deliberately has no repository root and no GitHub client. Callers
supply row, ownership, path, and collection facts so the same rules can be used
by offline checks, commands, migrations, and fixture tests.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import posixpath
import re
from typing import TypeAlias


FIELD_NAMES = (
    "ID",
    "Title",
    "Row",
    "State",
    "Kind",
    "GitHub",
    "Mirror",
    "Availability",
    "Created",
    "Updated",
    "Closed",
)
STATES = frozenset({"OPEN", "CLOSED", "UNKNOWN"})
MIRROR_STATES = frozenset({"SYNCED", "PENDING", "MISSING", "DIVERGED"})
AVAILABILITIES = frozenset({"FULL", "METADATA_ONLY"})

_GITHUB_ID = re.compile(r"ISSUE-GH-([1-9][0-9]*)\Z")
_LOCAL_ID = re.compile(r"ISSUE-LOCAL-([0-7][0-9A-HJKMNP-TV-Z]{25})\Z")
_ROW_ID = re.compile(r"[A-Z0-9][A-Za-z0-9_.-]*\Z")
_GITHUB_NUMBER = re.compile(r"[1-9][0-9]*\Z")
# The one frozen archive an intake record's Problem must quote, in
# repository-relative form. The record that HOLDS a quote lives deeper than
# the file the quote was cut from, which is why the comparison below
# resolves relative link targets instead of comparing their spelling: the
# same row is `../specs/x.md` in the archive and `../../specs/x.md` in the
# record, and both gates (this one and check-links) must be satisfiable at
# once.
FROZEN_ARCHIVE_RELPATH = ".agents/completed/issue-index.md"
_INTAKE_PROBLEM = re.compile(
    r"Archive: `" + re.escape(FROZEN_ARCHIVE_RELPATH) + r":([1-9][0-9]*)`\n\n"
    r"### Frozen archive evidence\n\n"
    r"> ([^\n]+)\Z"
)
_LINK_TARGET = re.compile(r"(\[[^\]]*\]\()([^)]+)(\))")
_REMOTE_TARGET = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


class IssueRecordError(ValueError):
    """A canonical issue record or reference violates the local schema."""


@dataclass(frozen=True, slots=True)
class IssueRecord:
    """The complete data stored in one canonical issue Markdown file."""

    id: str
    title: str
    row: str | None
    state: str
    kind: str
    github: int | None
    mirror: str
    availability: str
    created: str
    updated: str
    closed: str
    problem: str
    resolution: str


@dataclass(frozen=True, slots=True)
class GitHubProjection:
    """The only title, state, and body representation written to GitHub."""

    title: str
    state: str
    body: str


RowLookup: TypeAlias = Collection[str] | Callable[[str], bool]
OwedLookup: TypeAlias = (
    Collection[str]
    | Mapping[str, int | bool | Collection[object]]
    | Callable[[str], int | bool]
)


def _normal_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_body(text: str) -> str:
    """Normalize one Problem or Resolution section without adding a newline."""

    lines = [line.rstrip(" \t") for line in _normal_newlines(text).split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def normalize_github_body(text: str) -> str:
    """Normalize a complete readable GitHub body for exact comparison."""

    normalized = normalize_body(text)
    return f"{normalized}\n" if normalized else ""


def parse_issue_text(text: str) -> IssueRecord:
    """Parse one issue file's text and require the complete canonical shape."""

    text = _normal_newlines(text)
    problem_marker = "\n\n## Problem\n\n"
    resolution_marker = "\n\n## Resolution\n\n"
    if problem_marker not in text:
        raise IssueRecordError("issue file is missing the exact ## Problem heading")
    metadata, body = text.split(problem_marker, 1)
    if resolution_marker not in body:
        raise IssueRecordError("issue file is missing the exact ## Resolution heading")
    problem, resolution = body.split(resolution_marker, 1)

    lines = metadata.split("\n")
    if len(lines) != len(FIELD_NAMES):
        raise IssueRecordError("issue file must contain all canonical fields in schema order")

    values: dict[str, str] = {}
    for expected, line in zip(FIELD_NAMES, lines, strict=True):
        prefix = f"{expected}: "
        if not line.startswith(prefix):
            raise IssueRecordError("issue file must contain all canonical fields in schema order")
        value = line[len(prefix):]
        if not value:
            raise IssueRecordError(f"{expected} must not be empty")
        values[expected] = value

    github_text = values["GitHub"]
    if github_text == "-":
        github = None
    elif _GITHUB_NUMBER.fullmatch(github_text):
        github = int(github_text)
    else:
        raise IssueRecordError("GitHub must be a positive issue number or -")

    return IssueRecord(
        id=values["ID"],
        title=values["Title"],
        row=None if values["Row"] == "-" else values["Row"],
        state=values["State"],
        kind=values["Kind"],
        github=github,
        mirror=values["Mirror"],
        availability=values["Availability"],
        created=values["Created"],
        updated=values["Updated"],
        closed=values["Closed"],
        problem=problem,
        resolution=resolution.removesuffix("\n"),
    )


def parse_issue_file(path: str | Path) -> IssueRecord:
    """Read and parse one canonical issue file without inspecting the repository."""

    return parse_issue_text(Path(path).read_text(encoding="utf-8"))


def render_issue_record(record: IssueRecord) -> str:
    """Render one record with canonical field order, LF separators, and one newline."""

    problem = normalize_body(record.problem)
    resolution = normalize_body(record.resolution)
    github = "-" if record.github is None else str(record.github)
    row = "-" if record.row is None else record.row
    return (
        f"ID: {record.id}\n"
        f"Title: {record.title}\n"
        f"Row: {row}\n"
        f"State: {record.state}\n"
        f"Kind: {record.kind}\n"
        f"GitHub: {github}\n"
        f"Mirror: {record.mirror}\n"
        f"Availability: {record.availability}\n"
        f"Created: {record.created}\n"
        f"Updated: {record.updated}\n"
        f"Closed: {record.closed}\n"
        "\n## Problem\n\n"
        f"{problem}\n"
        "\n## Resolution\n\n"
        f"{resolution}\n"
    )


def issue_path(
    record: IssueRecord,
    issues_root: str | Path,
    *,
    intake: bool = False,
) -> Path:
    """Return the canonical path under a caller-supplied issue storage root."""

    if intake:
        owner = "_intake"
    else:
        owner = record.row if record.row is not None else "_owed"
    return Path(issues_root) / owner / f"{record.id}.md"

def canonical_rows(root: str | Path) -> set[str]:
    """Return claimable row identities from matrices, specs, and the roadmap."""

    root = Path(root)
    rows: set[str] = set()
    matrix_row = re.compile(r"\|\s*`([A-Z0-9][A-Za-z0-9_.-]*)`\s*\|")
    for path in (root / ".agents").glob("*-matrix.md"):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = matrix_row.match(line)
            if match:
                rows.add(match.group(1))

    declaration = re.compile(
        r"(?m)^(?:-\s+)?\*{0,2}"
        r"(?:Owning row|Row|Rows|Identity|Row slug):?\*{0,2}\s*"
        r"`([A-Z0-9][A-Za-z0-9_.-]*)`"
    )
    heading = re.compile(
        r"(?m)^#\s+(?:SPEC\s+—\s+)?`?([A-Z0-9][A-Za-z0-9_.-]*)`?"
        r"(?:\s+(?:—|-)|:)"
    )
    for path in (root / ".agents" / "specs").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        rows.update(match.group(1) for match in declaration.finditer(text))
        rows.update(match.group(1) for match in heading.finditer(text))

    roadmap = root / ".agents" / "roadmap_v1.md"
    if roadmap.is_file():
        roadmap_row = re.compile(
            r"(?m)^\|[^|\n]+\|\s*`([A-Z0-9][A-Za-z0-9_.-]*)`\s*\|"
        )
        rows.update(
            match.group(1)
            for match in roadmap_row.finditer(roadmap.read_text(encoding="utf-8"))
        )
    return rows


def owed_issue_counts(root: str | Path) -> Counter[str]:
    """Count the specs whose Owed section owns each stable issue identity."""

    counts: Counter[str] = Counter()
    stable = re.compile(
        r"(?<![A-Z0-9_-])"
        r"(ISSUE-GH-[1-9][0-9]*|ISSUE-LOCAL-[0-7][0-9A-HJKMNP-TV-Z]{25})"
        r"(?![A-Z0-9_-])"
    )
    for path in (Path(root) / ".agents" / "specs").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        section = re.search(r"(?ms)^## Owed\s*$\n(.*?)(?=^## |\Z)", text)
        if section is not None:
            counts.update(set(stable.findall(section.group(1))))
    return counts


def _is_real_date(value: str) -> bool:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _claimable(rows: RowLookup, row: str) -> bool:
    return bool(rows(row)) if callable(rows) else row in rows


def _owed_count(owed: OwedLookup, issue_id: str) -> int:
    if callable(owed):
        value = owed(issue_id)
        return int(value)
    if isinstance(owed, Mapping):
        value = owed.get(issue_id, 0)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        return len(value)
    if isinstance(owed, str):
        return int(owed == issue_id)
    return sum(candidate == issue_id for candidate in owed)


def _placeholder(value: str) -> bool:
    return normalize_body(value).strip() in {"", "-"}


def intake_archive_evidence(record: IssueRecord) -> tuple[int, str] | None:
    """Return the exact frozen archive line declared by an intake problem."""

    match = _INTAKE_PROBLEM.fullmatch(normalize_body(record.problem))
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _resolve_relative_links(line: str, base: str) -> str:
    """Rewrite every relative Markdown link target to the file it denotes.

    A relative link resolves against the file that QUOTES it, so one frozen
    row cannot keep one spelling in the archive (`.agents/completed/`) and in
    the record that quotes it (`.agents/issues/<owner>/`). Resolving both
    sides to the repository-relative path they point at is what lets the
    evidence comparison ask the question it means to ask -- is this the
    archived row? -- instead of which directory holds the quote.

    Remote (`https:`, `mailto:`, ...) and root-absolute (`/...`) targets are
    left exactly as written: only a rebase of a relative target is
    comparable, so a swapped remote URL, a truncated link, or any other
    difference still fails the comparison byte for byte.
    """

    def rewrite(match: re.Match[str]) -> str:
        raw = match.group(2)
        lead = raw[: len(raw) - len(raw.lstrip())]
        trail = raw[len(raw.rstrip()) :]
        target = raw.strip()
        angled = target.startswith("<") and target.endswith(">")
        if angled:
            target = target[1:-1]
        path, separator, fragment = target.partition("#")
        if (
            not path
            or path.startswith(("/", "?"))
            or _REMOTE_TARGET.match(path) is not None
        ):
            return match.group(0)
        resolved = posixpath.normpath(posixpath.join(base, path))
        rendered = f"{resolved}{separator}{fragment}"
        if angled:
            rendered = f"<{rendered}>"
        return f"{match.group(1)}{lead}{rendered}{trail}{match.group(3)}"

    return _LINK_TARGET.sub(rewrite, line)


def _archive_evidence_matches_source(
    evidence: tuple[int, str],
    frozen_archive: bytes | None,
    record_base: str = "",
) -> bool:
    """Require the declared line to be the quote, modulo relative link rebase.

    Byte equality is tried first and answers almost every record: an archive
    that has not moved and a record that copied the line verbatim need
    nothing resolved. Only when the bytes differ is the quote compared with
    each side's relative link targets resolved, which admits exactly the
    spelling a MOVE forces (`../specs/x.md` vs `../../specs/x.md`) and
    nothing else. `record_base` is the record's repository-relative
    directory; with no base to resolve against, only byte equality passes.
    """

    if frozen_archive is None:
        return False
    line_number, archived_line = evidence
    source_lines = frozen_archive.split(b"\n")
    if line_number > len(source_lines):
        return False
    source_line = source_lines[line_number - 1]
    if source_line == archived_line.encode("utf-8"):
        return True
    if not record_base:
        return False
    try:
        decoded = source_line.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return _resolve_relative_links(
        decoded, posixpath.dirname(FROZEN_ARCHIVE_RELPATH)
    ) == _resolve_relative_links(archived_line, record_base)


def _archive_row_owner(line: str, github: int | None) -> str | None:
    """Return the archived row cell when the evidence identifies this issue."""

    number = re.match(
        r"^\|\s*\[#([1-9][0-9]*)\]\(([^)]+)\)\s*\|",
        line,
    )
    if number is None or github is None or int(number.group(1)) != github:
        return None
    url_number = re.search(
        r"/issues/([1-9][0-9]*)(?:[/?#][^)\s]*)?\Z",
        number.group(2),
    )
    if url_number is None or int(url_number.group(1)) != github:
        return None
    cells = line[1:-1].split("|") if line.endswith("|") else line[1:].split("|")
    if len(cells) < 4:
        return None
    return cells[1].strip().strip("`")


def valid_intake_archive_evidence(
    record: IssueRecord,
    frozen_archive: bytes | None = None,
    record_base: str = "",
) -> tuple[int, str] | None:
    """Return evidence only when the source line identifies ownerless self.

    `record_base` is the record's repository-relative directory, so a quote
    whose links were re-pointed to resolve from there still matches the line
    it was cut from; see `_resolve_relative_links`.
    """

    evidence = intake_archive_evidence(record)
    if evidence is None or not _archive_evidence_matches_source(
        evidence,
        frozen_archive,
        record_base,
    ):
        return None
    _, line = evidence
    return evidence if _archive_row_owner(line, record.github) in {"", "-", "—"} else None


def validate_issue_record(
    record: IssueRecord,
    path: str | Path,
    rows: RowLookup,
    owed: OwedLookup,
    *,
    frozen_archive: bytes | None = None,
) -> None:
    """Validate schema, identity, state, and injected ownership facts.

    ``rows`` is either a collection of canonical claimable row IDs or a lookup
    callable. ``owed`` supplies the count of committed ``## Owed`` references:
    callers may pass an iterable of IDs, a count mapping, or a count callable.
    The function raises one ``IssueRecordError`` containing every found error.
    """

    errors: list[str] = []
    path = Path(path)

    github_id = _GITHUB_ID.fullmatch(record.id)
    local_id = _LOCAL_ID.fullmatch(record.id)
    if not github_id and not local_id:
        errors.append("ID must be ISSUE-GH-<number> or ISSUE-LOCAL-<ULID>")

    if path.name != f"{record.id}.md":
        errors.append(f"filename must equal issue identity {record.id}.md")

    if record.github is not None and (
        isinstance(record.github, bool) or not isinstance(record.github, int) or record.github <= 0
    ):
        errors.append("GitHub must be a positive issue number or absent")
    if github_id:
        id_number = int(github_id.group(1))
        if record.github != id_number:
            errors.append("ISSUE-GH identity and GitHub number must match exactly")

    if not record.title or record.title != record.title.strip() or "\n" in record.title or "\r" in record.title:
        errors.append("Title must be one nonempty line without surrounding whitespace")
    if (
        not record.kind
        or record.kind != record.kind.strip()
        or record.kind == "-"
        or "\n" in record.kind
        or "\r" in record.kind
    ):
        errors.append("Kind must be a nonempty issue kind or UNKNOWN")
    if record.row is not None and not _ROW_ID.fullmatch(record.row):
        errors.append("Row must be a canonical row ID or -")

    if record.state not in STATES:
        errors.append("State must be OPEN, CLOSED, or UNKNOWN")
    if record.mirror not in MIRROR_STATES:
        errors.append("Mirror must be SYNCED, PENDING, MISSING, or DIVERGED")
    if record.availability not in AVAILABILITIES:
        errors.append("Availability must be FULL or METADATA_ONLY")

    if record.github is None:
        if record.mirror != "PENDING":
            errors.append("Mirror must be PENDING without a GitHub number")
    elif record.mirror == "PENDING":
        errors.append("Mirror PENDING is invalid when a GitHub number exists")

    if record.state == "OPEN":
        if not _is_real_date(record.created):
            errors.append("Created must be a YYYY-MM-DD date for OPEN state")
        if not _is_real_date(record.updated):
            errors.append("Updated must be a YYYY-MM-DD date for OPEN state")
        if record.closed != "-":
            errors.append("Closed must be - for OPEN state")
    elif record.state == "CLOSED":
        if not _is_real_date(record.created):
            errors.append("Created must be a YYYY-MM-DD date for CLOSED state")
        if not _is_real_date(record.updated):
            errors.append("Updated must be a YYYY-MM-DD date for CLOSED state")
        if not _is_real_date(record.closed):
            errors.append("Closed must be a YYYY-MM-DD date for CLOSED state")
        if _placeholder(record.resolution):
            errors.append("Resolution evidence is required for CLOSED state")
    elif record.state == "UNKNOWN" and record.availability != "METADATA_ONLY":
        errors.append("State UNKNOWN is valid only with Availability METADATA_ONLY")

    if record.availability == "FULL":
        if _placeholder(record.problem):
            errors.append("Problem must contain canonical evidence for Availability FULL")
        if record.state == "UNKNOWN":
            errors.append("Availability FULL cannot claim State UNKNOWN")
    elif record.availability == "METADATA_ONLY":
        if record.state != "UNKNOWN":
            errors.append("Availability METADATA_ONLY requires State UNKNOWN")
        if record.mirror != "MISSING":
            errors.append("Availability METADATA_ONLY requires Mirror MISSING")
        for name, value in (
            ("Created", record.created),
            ("Updated", record.updated),
            ("Closed", record.closed),
        ):
            if value != "UNKNOWN":
                errors.append(f"{name} must be UNKNOWN for Availability METADATA_ONLY")
        if normalize_body(record.resolution).strip() != "-":
            errors.append("Resolution must be - for Availability METADATA_ONLY")
        archive = FROZEN_ARCHIVE_RELPATH
        exact_number = (
            record.github is not None
            and re.search(rf"(?<![0-9])#{record.github}(?![0-9])", record.problem)
        )
        if archive not in record.problem or not exact_number:
            errors.append("Problem must cite the exact frozen archive entry and GitHub number")

    issue_root = path.parent.parent
    if issue_root.name != "issues" or issue_root.parent.name != ".agents":
        errors.append(
            "file must use canonical issue path .agents/issues/<owner>/<ID>.md"
        )

    owner = path.parent.name
    # The record's own directory, repository-relative, so the frozen-evidence
    # comparison can resolve a re-pointed relative link against where the
    # quote lives instead of where the quote was cut from. Two levels under
    # .agents/, against the archive's one.
    record_base = f".agents/issues/{owner}"
    owed_count = _owed_count(owed, record.id)
    if owner == "_intake":
        intake_shape = (
            github_id is not None
            and record.row is None
            and record.state == "UNKNOWN"
            and record.mirror == "MISSING"
            and record.availability == "METADATA_ONLY"
            and record.created == "UNKNOWN"
            and record.updated == "UNKNOWN"
            and record.closed == "UNKNOWN"
            and normalize_body(record.resolution).strip() == "-"
        )
        if not intake_shape:
            errors.append(
                "_intake requires ISSUE-GH identity, Row -, UNKNOWN state and "
                "dates, MISSING mirror, METADATA_ONLY availability, and Resolution -"
            )
        evidence = intake_archive_evidence(record)
        if evidence is None:
            errors.append(
                "_intake Problem must use the exact Frozen archive evidence block"
            )
        else:
            _, archived_line = evidence
            archived_owner = _archive_row_owner(archived_line, record.github)
            if archived_owner not in {"", "-", "—"}:
                errors.append("_intake frozen evidence must contain archived Row -")
            if not _archive_evidence_matches_source(
                evidence, frozen_archive, record_base
            ):
                errors.append(
                    "_intake Frozen archive evidence must equal the declared line "
                    "in the frozen archive source (a relative link may differ only "
                    "by spelling, and must resolve to the same file)"
                )
        if owed_count:
            errors.append("_intake must not have an owning spec reference")
    elif owner == "_owed":
        if record.row is not None:
            errors.append("Row must be - for a file under _owed")
        if owed_count != 1:
            errors.append("an _owed issue must have exactly one owning spec reference")
    else:
        if record.row != owner:
            errors.append(f"path row {owner!r} must equal Row field {record.row or '-'}")
        elif not _claimable(rows, owner):
            errors.append(f"row {owner!r} is not canonical and claimable")
        if owed_count:
            errors.append("a row-owned issue must not retain an owed reference")

    if errors:
        raise IssueRecordError("; ".join(errors))


def validate_issue_collection(records: Iterable[IssueRecord]) -> None:
    """Require unique stable IDs and unique nonempty GitHub numbers."""

    materialized = list(records)
    ids = Counter(record.id for record in materialized)
    duplicate_ids = sorted(issue_id for issue_id, count in ids.items() if count > 1)
    github = Counter(record.github for record in materialized if record.github is not None)
    duplicate_github = sorted(number for number, count in github.items() if count > 1)
    errors: list[str] = []
    if duplicate_ids:
        errors.append(f"duplicate issue ID: {', '.join(duplicate_ids)}")
    if duplicate_github:
        errors.append(
            "duplicate GitHub number: " + ", ".join(f"#{number}" for number in duplicate_github)
        )
    if errors:
        raise IssueRecordError("; ".join(errors))


def resolve_issue_reference(
    reference: str, records: Iterable[IssueRecord]
) -> IssueRecord:
    """Resolve one stable local ID or ``#<number>`` to exactly one record."""

    materialized = list(records)
    if _GITHUB_ID.fullmatch(reference) or _LOCAL_ID.fullmatch(reference):
        matches = [record for record in materialized if record.id == reference]
    elif re.fullmatch(r"#[1-9][0-9]*", reference):
        number = int(reference[1:])
        matches = [record for record in materialized if record.github == number]
    else:
        raise IssueRecordError(f"issue reference {reference!r} cannot resolve: invalid syntax")
    if len(matches) != 1:
        raise IssueRecordError(
            f"issue reference {reference!r} must resolve to exactly one local record; found {len(matches)}"
        )
    return matches[0]


def github_projection(record: IssueRecord) -> GitHubProjection:
    """Project deterministic GitHub title, state, and normalized body."""

    if record.state == "UNKNOWN":
        raise IssueRecordError("State UNKNOWN has no GitHub state projection")
    if record.state not in {"OPEN", "CLOSED"}:
        raise IssueRecordError(f"State {record.state!r} has no GitHub state projection")
    problem = normalize_body(record.problem)
    if _placeholder(problem):
        raise IssueRecordError("Problem evidence is required for a GitHub projection")
    row = record.row if record.row is not None else "-"
    body = (
        f"Row: {row}\n"
        "\n"
        f"Local-Issue: {record.id}\n"
        f"Kind: {record.kind}\n"
        "\n"
        f"{problem}\n"
    )
    if record.state == "CLOSED":
        resolution = normalize_body(record.resolution)
        if _placeholder(resolution):
            raise IssueRecordError("Resolution evidence is required for a CLOSED projection")
        body += f"\n## Resolution\n\n{resolution}\n"
    return GitHubProjection(title=record.title, state=record.state, body=body)
