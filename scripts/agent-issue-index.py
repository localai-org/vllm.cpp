#!/usr/bin/env python3
"""Render the untracked issue view from canonical local issue records."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Collection
from pathlib import Path

import issue_records


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"
ISSUES_ROOT = AGENTS / "issues"
FROZEN_ARCHIVE = AGENTS / "completed" / "issue-index.md"
SNAPSHOT = AGENTS / "issue-index.generated.md"
REPO_URL = "https://github.com/mudler/vllm.cpp"

PREAMBLE = """# Issue index (GENERATED -- do not edit, do not commit)

Canonical authority lives in the tracked files under `.agents/issues/`.
Regenerate this convenience view with
`python3 scripts/agent-issue-index.py --refresh`.

| Issue | Row | Title | Kind |
|---:|---|---|---|"""


def _markdown_table_cell(value: str) -> str:
    """Render one free-form value without adding Markdown table cells."""

    return " ".join(value.split()).replace("\\", "\\\\").replace("|", "\\|")


def render_local(records: list[issue_records.IssueRecord]) -> str:
    """Render open canonical records in stable identity order."""

    lines = [PREAMBLE]
    ordered = sorted(
        (record for record in records if record.state == "OPEN"),
        key=lambda record: (
            record.github is None,
            record.github if record.github is not None else record.id,
        ),
    )
    for record in ordered:
        title = _markdown_table_cell(record.title)
        kind = _markdown_table_cell(record.kind)
        row = f"`{record.row}`" if record.row else "—"
        identity = (
            f"[#{record.github}]({REPO_URL}/issues/{record.github})"
            if record.github is not None
            else f"`{record.id}`"
        )
        lines.append(f"| {identity} | {row} | {title} | {kind} |")
    return "\n".join(lines) + "\n"


def load_local_files(
    issues_root: Path = ISSUES_ROOT,
    *,
    rows: Collection[str] | None = None,
    owed: issue_records.OwedLookup | None = None,
    frozen_archive: bytes | None = None,
) -> list[issue_records.IssueRecord]:
    """Parse and validate every canonical issue file without network access.

    Every invalid file is reported in one run. Stopping at the first failure
    is what let three separate defects hide behind one another in the
    ORPHAN-MODEL-ROWS repair (ISSUE-LOCAL-01M3NC14GE995V9E6F7GTYSQJ3): each
    fix exposed the next only after another run, and the first failure was
    always an _intake record nobody was editing.
    """

    effective_rows = issue_records.canonical_rows(ROOT) if rows is None else rows
    effective_owed = issue_records.owed_issue_counts(ROOT) if owed is None else owed
    if frozen_archive is None:
        frozen_archive = FROZEN_ARCHIVE.read_bytes()
    records: list[issue_records.IssueRecord] = []
    failures: list[str] = []
    for path in sorted(issues_root.glob("**/*.md")):
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
            failures.append(f"{path}: {error}")
            continue
        records.append(record)
    if failures:
        raise issue_records.IssueRecordError("; ".join(failures))
    issue_records.validate_issue_collection(records)
    return records


def render_local_files(issues_root: Path = ISSUES_ROOT) -> str:
    """Parse and render local files for fixture and preview callers."""

    records = [
        issue_records.parse_issue_file(path)
        for path in sorted(issues_root.glob("**/*.md"))
    ]
    issue_records.validate_issue_collection(records)
    return render_local(records)


def refresh(
    *,
    rows: Collection[str] | None = None,
    owed: issue_records.OwedLookup | None = None,
) -> tuple[int, str]:
    """Validate local authority and rewrite its untracked convenience view."""

    try:
        records = load_local_files(ISSUES_ROOT, rows=rows, owed=owed)
        text = render_local(records)
        SNAPSHOT.write_text(text, encoding="utf-8")
    except (OSError, issue_records.IssueRecordError) as error:
        return 2, f"canonical issue refresh failed: {error}"

    open_count = sum(record.state == "OPEN" for record in records)
    label = SNAPSHOT.relative_to(ROOT) if SNAPSHOT.is_relative_to(ROOT) else SNAPSHOT
    return 0, (
        f"wrote {label} from {len(records)} canonical issue records: "
        f"{open_count} open canonical issue{'s' if open_count != 1 else ''}"
    )


def snapshot_state() -> tuple[str | None, str | None]:
    """Return the generated view and a reason when it differs from authority."""

    try:
        expected = render_local(load_local_files())
    except (OSError, issue_records.IssueRecordError) as error:
        return None, f"canonical issue files are invalid: {error}"
    if not SNAPSHOT.is_file():
        return None, "run `python3 scripts/agent-issue-index.py --refresh`"
    try:
        actual = SNAPSHOT.read_text(encoding="utf-8")
    except OSError as error:
        return None, f"cannot read {SNAPSHOT.relative_to(ROOT)}: {error}"
    if actual != expected:
        return actual, "generated view differs; run `python3 scripts/agent-issue-index.py --refresh`"
    return actual, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="validate canonical local records and rewrite the generated view",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the generated view against canonical local records",
    )
    parser.add_argument(
        "--render-local",
        metavar="ISSUES_ROOT",
        help="preview the local canonical renderer on an issue directory",
    )
    arguments = parser.parse_args(argv)

    if arguments.render_local:
        print(render_local_files(Path(arguments.render_local)), end="")
        return 0
    if arguments.refresh:
        code, message = refresh()
        print(message, file=sys.stderr if code else sys.stdout)
        return code
    if arguments.check:
        _, reason = snapshot_state()
        if reason is not None:
            print(f"ERROR: {reason}", file=sys.stderr)
            return 2
        print(f"OK: {SNAPSHOT.relative_to(ROOT)} matches canonical issue files")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
