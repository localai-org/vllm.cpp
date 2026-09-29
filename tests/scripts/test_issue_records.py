#!/usr/bin/env python3
"""Focused tests for canonical local issue record parsing and validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import issue_records as records


LOCAL_ID = f"ISSUE-LOCAL-{''}01ARZ3NDEKTSV4RRFFQ69G5FAV"


def issue_text(**changes: str) -> str:
    values = {
        "ID": f"ISSUE-GH-{''}42",
        "Title": "Repair record authority",
        "Row": "ROW-A",
        "State": "OPEN",
        "Kind": "bug",
        "GitHub": "42",
        "Mirror": "SYNCED",
        "Availability": "FULL",
        "Created": "2026-08-01",
        "Updated": "2026-08-31",
        "Closed": "-",
        "Problem": "The local record must remain authoritative.",
        "Resolution": "-",
    }
    values.update(changes)
    return (
        f"ID: {values['ID']}\n"
        f"Title: {values['Title']}\n"
        f"Row: {values['Row']}\n"
        f"State: {values['State']}\n"
        f"Kind: {values['Kind']}\n"
        f"GitHub: {values['GitHub']}\n"
        f"Mirror: {values['Mirror']}\n"
        f"Availability: {values['Availability']}\n"
        f"Created: {values['Created']}\n"
        f"Updated: {values['Updated']}\n"
        f"Closed: {values['Closed']}\n"
        "\n## Problem\n\n"
        f"{values['Problem']}\n"
        "\n## Resolution\n\n"
        f"{values['Resolution']}\n"
    )


def intake_problem(number: int = 77, line: int = 6, row: str = "—") -> str:
    archived = (
        f"| [#{number}](https://github.com/mudler/vllm.cpp/issues/{number}) "
        f"| {row} | Archived {number} | bug |"
    )
    return (
        f"Archive: `.agents/completed/issue-index.md:{line}`\n\n"
        "### Frozen archive evidence\n\n"
        f"> {archived}"
    )


def frozen_archive_source(
    number: int = 77,
    *,
    row: str = "—",
    title: str | None = None,
    kind: str = "bug",
    link_number: int | None = None,
) -> bytes:
    archived_title = title or f"Archived {number}"
    target = number if link_number is None else link_number
    archived = (
        f"| [#{number}](https://github.com/mudler/vllm.cpp/issues/{target}) "
        f"| {row} | {archived_title} | {kind} |"
    )
    return (
        "# Issue index\n\nFrozen archive\n"
        "| Issue | Row | Title | Kind |\n"
        "|---:|---|---|---|\n"
        f"{archived}\n"
    ).encode()


def linked_intake_problem(
    number: int = 77,
    line: int = 6,
    spec: str = "../specs/example.md",
) -> str:
    """An intake Problem whose quote carries a relative link, archive-spelled.

    The archive lives at `.agents/completed/issue-index.md`, so a link to a
    spec is `../specs/...` there. The record that QUOTES the row lives two
    levels down, and the same link is `../../specs/...` from there -- the
    spelling #3348 is about.
    """
    archived = (
        f"| [#{number}](https://github.com/mudler/vllm.cpp/issues/{number}) "
        f"| — | Spec [example]({spec}) | bug |"
    )
    return (
        f"Archive: `.agents/completed/issue-index.md:{line}`\n\n"
        "### Frozen archive evidence\n\n"
        f"> {archived}"
    )


def linked_frozen_archive_source(number: int = 77, spec: str = "../specs/example.md") -> bytes:
    archived = (
        f"| [#{number}](https://github.com/mudler/vllm.cpp/issues/{number}) "
        f"| — | Spec [example]({spec}) | bug |"
    )
    return (
        "# Issue index\n\nFrozen archive\n"
        "| Issue | Row | Title | Kind |\n"
        "|---:|---|---|---|\n"
        f"{archived}\n"
    ).encode()


def parse(text: str | None = None) -> records.IssueRecord:
    return records.parse_issue_text(text if text is not None else issue_text())


def path_for(tmp_path: Path, record: records.IssueRecord, owner: str = "ROW-A") -> Path:
    return tmp_path / ".agents" / "issues" / owner / f"{record.id}.md"


def validate(
    tmp_path: Path,
    record: records.IssueRecord,
    *,
    owner: str = "ROW-A",
    rows=frozenset({"ROW-A"}),
    owed=(),
    frozen_archive: bytes | None = None,
) -> None:
    records.validate_issue_record(
        record,
        path_for(tmp_path, record, owner),
        rows,
        owed,
        frozen_archive=frozen_archive,
    )


class TestParsingAndRendering:
    def test_parse_file_reads_every_canonical_field_and_body_section(self, tmp_path: Path) -> None:
        path = tmp_path / "record.md"
        path.write_text(issue_text(), encoding="utf-8")

        record = records.parse_issue_file(path)

        assert record == records.IssueRecord(
            id=f"ISSUE-GH-{''}42",
            title="Repair record authority",
            row="ROW-A",
            state="OPEN",
            kind="bug",
            github=42,
            mirror="SYNCED",
            availability="FULL",
            created="2026-08-01",
            updated="2026-08-31",
            closed="-",
            problem="The local record must remain authoritative.",
            resolution="-",
        )

    @pytest.mark.parametrize(
        "line",
        [
            f"ID: ISSUE-GH-{''}42\n",
            "Title: Repair record authority\n",
            "Row: ROW-A\n",
            "State: OPEN\n",
            "Kind: bug\n",
            "GitHub: 42\n",
            "Mirror: SYNCED\n",
            "Availability: FULL\n",
            "Created: 2026-08-01\n",
            "Updated: 2026-08-31\n",
            "Closed: -\n",
        ],
    )
    def test_parser_rejects_each_missing_required_field(self, line: str) -> None:
        with pytest.raises(records.IssueRecordError, match="canonical fields"):
            records.parse_issue_text(issue_text().replace(line, "", 1))

    @pytest.mark.parametrize("heading", ["## Problem", "## Resolution"])
    def test_parser_rejects_each_missing_required_heading(self, heading: str) -> None:
        with pytest.raises(records.IssueRecordError, match="heading"):
            records.parse_issue_text(issue_text().replace(heading, "Missing", 1))

    def test_parser_rejects_unknown_or_reordered_metadata(self) -> None:
        text = issue_text().replace("Title: Repair record authority\n", "Extra: value\nTitle: Repair record authority\n")
        with pytest.raises(records.IssueRecordError, match="canonical fields"):
            records.parse_issue_text(text)

    def test_render_is_canonical_and_round_trips(self) -> None:
        record = replace(
            parse(),
            problem="\r\nFirst line  \rSecond\t\r\n\r\n",
            resolution="  -  \r\n",
        )
        expected = issue_text(Problem="First line\nSecond", Resolution="  -")

        rendered = records.render_issue_record(record)

        assert rendered == expected
        assert rendered.endswith("\n") and not rendered.endswith("\n\n")
        assert records.parse_issue_text(rendered) == replace(
            record, problem="First line\nSecond", resolution="  -"
        )


class TestRecordValidation:
    def test_valid_open_full_record(self, tmp_path: Path) -> None:
        validate(tmp_path, parse())

    @pytest.mark.parametrize("field", ["created", "updated"])
    @pytest.mark.parametrize("value", ["UNKNOWN", "-", "2026-02-30", "2026-8-01"])
    def test_open_requires_real_iso_dates(self, tmp_path: Path, field: str, value: str) -> None:
        with pytest.raises(records.IssueRecordError, match=field.capitalize()):
            validate(tmp_path, replace(parse(), **{field: value}))

    @pytest.mark.parametrize("closed", ["UNKNOWN", "2026-08-31"])
    def test_open_requires_dash_closed_date(self, tmp_path: Path, closed: str) -> None:
        with pytest.raises(records.IssueRecordError, match="Closed"):
            validate(tmp_path, replace(parse(), closed=closed))

    def test_closed_requires_dates_and_resolution_evidence(self, tmp_path: Path) -> None:
        closed = replace(parse(), state="CLOSED", closed="2026-08-31", resolution="Fixed by abc123.")
        validate(tmp_path, closed)

        for field, value in (("created", "UNKNOWN"), ("updated", "-"), ("closed", "UNKNOWN"), ("resolution", "-")):
            with pytest.raises(records.IssueRecordError):
                validate(tmp_path, replace(closed, **{field: value}))

    def test_kind_unknown_is_valid_without_inference(self, tmp_path: Path) -> None:
        validate(tmp_path, replace(parse(), kind="UNKNOWN"))

    def test_kind_must_remain_one_canonical_line(self, tmp_path: Path) -> None:
        with pytest.raises(records.IssueRecordError, match="Kind"):
            validate(tmp_path, replace(parse(), kind="bug\rInjected: value"))


    @pytest.mark.parametrize("state", ["ACTIVE", "open", ""])
    def test_only_canonical_states_are_valid(self, tmp_path: Path, state: str) -> None:
        with pytest.raises(records.IssueRecordError, match="State"):
            validate(tmp_path, replace(parse(), state=state))

    @pytest.mark.parametrize("availability", ["PARTIAL", "full", ""])
    def test_only_canonical_availability_values_are_valid(self, tmp_path: Path, availability: str) -> None:
        with pytest.raises(records.IssueRecordError, match="Availability"):
            validate(tmp_path, replace(parse(), availability=availability))

    def test_full_record_requires_a_real_problem(self, tmp_path: Path) -> None:
        for problem in ("", "-", " \n\t"):
            with pytest.raises(records.IssueRecordError, match="Problem"):
                validate(tmp_path, replace(parse(), problem=problem))

    def test_unknown_metadata_only_archive_placeholder_is_valid(self, tmp_path: Path) -> None:
        record = parse(
            issue_text(
                ID=f"ISSUE-GH-{''}77",
                GitHub="77",
                State="UNKNOWN",
                Mirror="MISSING",
                Availability="METADATA_ONLY",
                Created="UNKNOWN",
                Updated="UNKNOWN",
                Closed="UNKNOWN",
                Problem="Frozen archive entry: .agents/completed/issue-index.md row for #77.",
            )
        )
        validate(tmp_path, record)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("state", "OPEN"),
            ("mirror", "SYNCED"),
            ("created", "2026-08-01"),
            ("updated", "2026-08-01"),
            ("closed", "-"),
            ("resolution", "Inferred from archive."),
            ("problem", "Archive identity existed."),
        ],
    )
    def test_metadata_only_does_not_infer_remote_facts(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        record = parse(
            issue_text(
                ID=f"ISSUE-GH-{''}77",
                GitHub="77",
                State="UNKNOWN",
                Mirror="MISSING",
                Availability="METADATA_ONLY",
                Created="UNKNOWN",
                Updated="UNKNOWN",
                Closed="UNKNOWN",
                Problem="Frozen archive entry: .agents/completed/issue-index.md row for #77.",
            )
        )
        with pytest.raises(records.IssueRecordError):
            validate(tmp_path, replace(record, **{field: value}))

    @pytest.mark.parametrize(
        "problem",
        [
            "Frozen archive entry: .agents/completed/issue-index.md row for #770.",
            "Frozen archive entry: another-index.md row for #77.",
        ],
    )
    def test_metadata_only_citation_matches_its_exact_archive_identity(
        self, tmp_path: Path, problem: str
    ) -> None:
        record = parse(
            issue_text(
                ID=f"ISSUE-GH-{''}77",
                GitHub="77",
                State="UNKNOWN",
                Mirror="MISSING",
                Availability="METADATA_ONLY",
                Created="UNKNOWN",
                Updated="UNKNOWN",
                Closed="UNKNOWN",
                Problem=problem,
            )
        )
        with pytest.raises(records.IssueRecordError, match="exact frozen archive"):
            validate(tmp_path, record)


    def test_unknown_state_is_rejected_for_full_records(self, tmp_path: Path) -> None:
        with pytest.raises(records.IssueRecordError, match="UNKNOWN.*METADATA_ONLY"):
            validate(
                tmp_path,
                replace(parse(), state="UNKNOWN", created="UNKNOWN", updated="UNKNOWN", closed="UNKNOWN"),
            )


class TestIdentityMirrorAndOwnership:
    def test_github_id_number_and_filename_must_match(self, tmp_path: Path) -> None:
        record = parse()
        cases = [
            (replace(record, github=43), path_for(tmp_path, record)),
            (record, path_for(tmp_path, record).with_name(f"ISSUE-GH-{''}43.md")),
            (replace(record, id=f"ISSUE-GH-{''}43"), path_for(tmp_path, record)),
        ]
        for candidate, path in cases:
            with pytest.raises(records.IssueRecordError, match="identity|filename|GitHub"):
                records.validate_issue_record(candidate, path, {"ROW-A"}, ())

    @pytest.mark.parametrize(
        "issue_id",
        [
            "ISSUE-GH-0",
            "ISSUE-GH-042",
            "ISSUE-LOCAL-not-a-ulid",
            "ISSUE-LOCAL-01ARZ3NDEKTSV4RRFFQ69G5FAI",
            "ISSUE-LOCAL-ZZZZZZZZZZZZZZZZZZZZZZZZZZ",
        ],
    )
    def test_only_canonical_hybrid_ids_are_valid(
        self, tmp_path: Path, issue_id: str
    ) -> None:
        record = replace(parse(), id=issue_id)
        with pytest.raises(records.IssueRecordError, match="ID must"):
            records.validate_issue_record(
                record,
                path_for(tmp_path, record),
                {"ROW-A"},
                (),
            )

    def test_local_id_filename_survives_row_assignment_and_mirroring(self, tmp_path: Path) -> None:
        record = parse(issue_text(ID=LOCAL_ID, GitHub="501", Mirror="DIVERGED"))
        validate(tmp_path, record)
        assert path_for(tmp_path, record).name == f"{LOCAL_ID}.md"

    @pytest.mark.parametrize("mirror", ["SYNCED", "MISSING", "DIVERGED"])
    def test_numbered_mirror_states_require_a_github_number(self, tmp_path: Path, mirror: str) -> None:
        record = parse(issue_text(ID=LOCAL_ID, GitHub="-", Mirror=mirror))
        with pytest.raises(records.IssueRecordError, match="Mirror"):
            validate(tmp_path, record)

    def test_record_without_github_number_must_be_pending(self, tmp_path: Path) -> None:
        record = parse(issue_text(ID=LOCAL_ID, GitHub="-", Mirror="PENDING"))
        validate(tmp_path, record)

    def test_numbered_record_cannot_be_pending(self, tmp_path: Path) -> None:
        with pytest.raises(records.IssueRecordError, match="Mirror"):
            validate(tmp_path, replace(parse(), mirror="PENDING"))

    def test_row_directory_must_equal_row_field(self, tmp_path: Path) -> None:
        with pytest.raises(records.IssueRecordError, match="path row"):
            validate(tmp_path, parse(), owner="ROW-B", rows={"ROW-A", "ROW-B"})

    def test_nested_issue_directory_is_not_a_canonical_path(self, tmp_path: Path) -> None:
        record = parse()
        path = (
            tmp_path
            / ".agents"
            / "issues"
            / "archive"
            / "ROW-A"
            / f"{record.id}.md"
        )
        with pytest.raises(records.IssueRecordError, match="canonical issue path"):
            records.validate_issue_record(record, path, {"ROW-A"}, ())

    def test_row_must_be_canonical_and_claimable_via_injected_lookup(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def is_claimable(row: str) -> bool:
            seen.append(row)
            return False

        with pytest.raises(records.IssueRecordError, match="claimable"):
            validate(tmp_path, parse(), rows=is_claimable)
        assert seen == ["ROW-A"]

    def test_row_identity_accepts_the_mixed_case_ids_used_by_model_matrices(
        self,
        tmp_path: Path,
    ) -> None:
        row = "MODEL-MM-qwen4-exp"
        record = replace(parse(), row=row)

        validate(tmp_path, record, owner=row, rows={row})

    def test_owed_path_requires_dash_row_and_one_injected_owner(self, tmp_path: Path) -> None:
        record = parse(issue_text(ID=LOCAL_ID, Row="-", GitHub="-", Mirror="PENDING"))
        validate(tmp_path, record, owner="_owed", rows=set(), owed=[LOCAL_ID])

        for owed in ([], [LOCAL_ID, LOCAL_ID]):
            with pytest.raises(records.IssueRecordError, match="exactly one"):
                validate(tmp_path, record, owner="_owed", rows=set(), owed=owed)
        with pytest.raises(records.IssueRecordError, match="Row"):
            validate(tmp_path, replace(record, row="ROW-A"), owner="_owed", owed=[LOCAL_ID])

    def test_row_owned_record_rejects_stale_owed_reference(self, tmp_path: Path) -> None:
        with pytest.raises(records.IssueRecordError, match="owed"):
            validate(tmp_path, parse(), owed=[f"ISSUE-GH-{''}42"])

    def test_expected_path_uses_injected_root_and_owner(self, tmp_path: Path) -> None:
        root = tmp_path / "issues"
        assert records.issue_path(parse(), root) == root / "ROW-A" / f"ISSUE-GH-{''}42.md"
        owed = parse(issue_text(ID=LOCAL_ID, Row="-", GitHub="-", Mirror="PENDING"))
        assert records.issue_path(owed, root) == root / "_owed" / f"{LOCAL_ID}.md"


class TestMigrationIntake:
    def intake(self, **changes: object) -> records.IssueRecord:
        values: dict[str, object] = {
            "id": f"ISSUE-GH-{''}77",
            "title": "Archived 77",
            "row": None,
            "state": "UNKNOWN",
            "kind": "bug",
            "github": 77,
            "mirror": "MISSING",
            "availability": "METADATA_ONLY",
            "created": "UNKNOWN",
            "updated": "UNKNOWN",
            "closed": "UNKNOWN",
            "problem": intake_problem(),
            "resolution": "-",
        }
        values.update(changes)
        return records.IssueRecord(**values)

    def test_intake_accepts_only_the_fixed_ownerless_archive_shape(
        self, tmp_path: Path
    ) -> None:
        record = self.intake()
        validate(
            tmp_path,
            record,
            owner="_intake",
            rows={"ROW-A"},
            owed=(),
            frozen_archive=frozen_archive_source(),
        )
        assert records.issue_path(record, tmp_path / "issues", intake=True) == (
            tmp_path / "issues" / "_intake" / f"ISSUE-GH-{''}77.md"
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("id", LOCAL_ID),
            ("row", "ROW-A"),
            ("state", "OPEN"),
            ("mirror", "DIVERGED"),
            ("availability", "FULL"),
            ("created", "2026-08-01"),
            ("updated", "2026-08-01"),
            ("closed", "-"),
            ("resolution", "Invented."),
            ("resolution", ""),
        ],
    )
    def test_intake_rejects_every_nonarchive_schema_value(
        self, tmp_path: Path, field: str, value: object
    ) -> None:
        record = self.intake(**{field: value})
        with pytest.raises(records.IssueRecordError, match="_intake"):
            validate(tmp_path, record, owner="_intake", rows={"ROW-A"}, owed=())

    @pytest.mark.parametrize(
        "problem",
        [
            "Archive: `.agents/completed/issue-index.md:6`\n\n> row",
            intake_problem().replace(":6`", ":line`"),
            intake_problem().replace("### Frozen archive evidence", "### Evidence"),
            intake_problem().replace("\n\n> |", "\n> |"),
            intake_problem() + "\n> another row",
            intake_problem(row="`ROW-A`"),
        ],
    )
    def test_intake_requires_exact_frozen_archive_evidence(
        self, tmp_path: Path, problem: str
    ) -> None:
        with pytest.raises(records.IssueRecordError, match="Frozen archive evidence|archived Row"):
            validate(
                tmp_path,
                self.intake(problem=problem),
                owner="_intake",
                rows={"ROW-A"},
                owed=(),
                frozen_archive=frozen_archive_source(),
            )

    @pytest.mark.parametrize(
        ("problem", "frozen_archive"),
        [
            (intake_problem(line=5), frozen_archive_source()),
            (intake_problem(line=99), frozen_archive_source()),
            (
                intake_problem().replace("/issues/77)", "/issues/78)"),
                frozen_archive_source(),
            ),
            (intake_problem(row="-"), frozen_archive_source()),
            (
                intake_problem().replace("Archived 77", "Changed title"),
                frozen_archive_source(),
            ),
            (
                intake_problem().replace("| bug |", "| feature |"),
                frozen_archive_source(),
            ),
            (intake_problem(), b"# Issue index\n"),
            (intake_problem(), frozen_archive_source(78)),
        ],
    )
    def test_intake_evidence_must_equal_the_declared_frozen_source_line(
        self,
        tmp_path: Path,
        problem: str,
        frozen_archive: bytes,
    ) -> None:
        with pytest.raises(records.IssueRecordError, match="frozen archive source"):
            validate(
                tmp_path,
                self.intake(problem=problem),
                owner="_intake",
                rows={"ROW-A"},
                owed=(),
                frozen_archive=frozen_archive,
            )

    def test_intake_quote_may_rebase_a_relative_link_to_resolve_from_the_record(
        self, tmp_path: Path
    ) -> None:
        """The record-relative spelling of a link is the SAME link (#3348).

        `check-links` requires every link in a record to resolve from the
        record's own directory; the frozen-evidence comparison required the
        quote to be byte-identical to the archive line it was cut from. One
        string could not satisfy both, because the archive is one level under
        `.agents/` and the record two. A quote whose ONLY difference is a
        relative link resolving to the same file is accepted.
        """
        record = self.intake(problem=linked_intake_problem(spec="../../specs/example.md"))
        validate(
            tmp_path,
            record,
            owner="_intake",
            rows={"ROW-A"},
            owed=(),
            frozen_archive=linked_frozen_archive_source(),
        )

    @pytest.mark.parametrize(
        ("spec", "archive_spec"),
        [
            # a rebase to a DIFFERENT file is still drift
            ("../../specs/other.md", "../specs/example.md"),
            ("../../specs/example.md#frag", "../specs/example.md"),
            # a remote target is never rebased, so a swap must fail byte for byte
            ("https://example.com/example.md", "../specs/example.md"),
            # a root-absolute target is never rebased either
            ("/specs/example.md", "../specs/example.md"),
        ],
    )
    def test_intake_quote_may_rebase_nothing_but_a_relative_link(
        self, tmp_path: Path, spec: str, archive_spec: str
    ) -> None:
        record = self.intake(problem=linked_intake_problem(spec=spec))
        with pytest.raises(records.IssueRecordError, match="frozen archive source"):
            validate(
                tmp_path,
                record,
                owner="_intake",
                rows={"ROW-A"},
                owed=(),
                frozen_archive=linked_frozen_archive_source(spec=archive_spec),
            )

    def test_a_rebased_quote_needs_a_record_base_to_be_compared(self) -> None:
        """Without the record's directory, only byte equality may pass.

        The public helper keeps the old strict behaviour for a caller that
        cannot say where the quote lives, rather than guessing a base and
        admitting a drift it cannot see.
        """
        record = records.IssueRecord(
            id="ISSUE-GH-77",
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
            problem=linked_intake_problem(spec="../../specs/example.md"),
            resolution="-",
        )
        assert records.valid_intake_archive_evidence(
            record, linked_frozen_archive_source()
        ) is None
        assert records.valid_intake_archive_evidence(
            record, linked_frozen_archive_source(), ".agents/issues/_intake"
        ) is not None

    def test_full_and_local_records_cannot_use_intake(self, tmp_path: Path) -> None:
        for record in (
            parse(),
            parse(issue_text(ID=LOCAL_ID, GitHub="-", Mirror="PENDING")),
        ):
            with pytest.raises(records.IssueRecordError, match="_intake"):
                validate(tmp_path, record, owner="_intake", rows={"ROW-A"}, owed=())


class TestCollectionsAndReferences:
    def test_collection_rejects_duplicate_local_ids(self) -> None:
        one = parse()
        with pytest.raises(records.IssueRecordError, match="duplicate issue ID"):
            records.validate_issue_collection([one, replace(one, github=43)])

    def test_collection_rejects_duplicate_nonempty_github_numbers(self) -> None:
        one = parse()
        two = parse(issue_text(ID=LOCAL_ID, GitHub="42", Mirror="DIVERGED"))
        with pytest.raises(records.IssueRecordError, match="duplicate GitHub"):
            records.validate_issue_collection([one, two])

    def test_collection_allows_multiple_records_without_github_numbers(self) -> None:
        one = parse(issue_text(ID=LOCAL_ID, GitHub="-", Mirror="PENDING"))
        two = replace(one, id=f"ISSUE-LOCAL-{''}01BX5ZZKBKACTAV9WEVGEMMVRZ")
        records.validate_issue_collection([one, two])

    def test_reference_resolves_stable_id_or_github_number(self) -> None:
        gh = parse()
        local = parse(issue_text(ID=LOCAL_ID, GitHub="501", Mirror="DIVERGED"))
        collection = [gh, local]
        assert records.resolve_issue_reference(f"ISSUE-GH-{''}42", collection) is gh
        assert records.resolve_issue_reference(LOCAL_ID, collection) is local
        assert records.resolve_issue_reference("#501", collection) is local

    @pytest.mark.parametrize("reference", [f"ISSUE-GH-{''}999", "#999", "GH-42"])
    def test_reference_must_resolve_to_exactly_one_record(self, reference: str) -> None:
        with pytest.raises(records.IssueRecordError, match="resolve"):
            records.resolve_issue_reference(reference, [parse()])

    def test_duplicate_reference_is_rejected_even_before_collection_validation(self) -> None:
        one = parse()
        two = parse(issue_text(ID=LOCAL_ID, GitHub="42", Mirror="DIVERGED"))
        with pytest.raises(records.IssueRecordError, match="exactly one"):
            records.resolve_issue_reference("#42", [one, two])


class TestGitHubProjection:
    def test_open_projection_normalizes_exact_body_shape(self) -> None:
        record = replace(
            parse(),
            problem="\r\nFirst line  \r\n\r\nSecond\t\r\n\r\n",
        )

        projection = records.github_projection(record)

        assert projection == records.GitHubProjection(
            title="Repair record authority",
            state="OPEN",
            body=(
                "Row: ROW-A\n"
                "\n"
                f"Local-Issue: ISSUE-GH-{''}42\n"
                "Kind: bug\n"
                "\n"
                "First line\n"
                "\n"
                "Second\n"
            ),
        )

    def test_closed_projection_appends_normalized_resolution(self) -> None:
        record = replace(
            parse(),
            state="CLOSED",
            closed="2026-08-31",
            problem=" Problem\t\n",
            resolution="\nFixed here.  \r\n\r\nEvidence.\t\n",
        )
        assert records.github_projection(record).body == (
            "Row: ROW-A\n"
            "\n"
            f"Local-Issue: ISSUE-GH-{''}42\n"
            "Kind: bug\n"
            "\n"
            " Problem\n"
            "\n"
            "## Resolution\n"
            "\n"
            "Fixed here.\n"
            "\n"
            "Evidence.\n"
        )

    def test_open_projection_does_not_append_resolution(self) -> None:
        body = records.github_projection(replace(parse(), resolution="Work in progress.")).body
        assert "## Resolution" not in body

    def test_dash_row_projects_as_one_dash(self) -> None:
        record = parse(issue_text(ID=LOCAL_ID, Row="-", GitHub="-", Mirror="PENDING"))
        assert records.github_projection(record).body.startswith("Row: -\n\n")

    def test_unknown_state_has_no_github_projection(self) -> None:
        record = replace(
            parse(),
            state="UNKNOWN",
            mirror="MISSING",
            availability="METADATA_ONLY",
            created="UNKNOWN",
            updated="UNKNOWN",
            closed="UNKNOWN",
        )
        with pytest.raises(records.IssueRecordError, match="UNKNOWN"):
            records.github_projection(record)

    def test_body_normalization_preserves_internal_blank_lines_only(self) -> None:
        assert records.normalize_body("\r\n\t\r\nA  \rB\t\n\n\t\n") == "A\nB"

    def test_remote_body_normalization_has_lf_and_one_final_newline(self) -> None:
        body = "\r\nRow: ROW-A  \r\n\r\nProblem\t\r\n\r\n"
        assert records.normalize_github_body(body) == "Row: ROW-A\n\nProblem\n"
