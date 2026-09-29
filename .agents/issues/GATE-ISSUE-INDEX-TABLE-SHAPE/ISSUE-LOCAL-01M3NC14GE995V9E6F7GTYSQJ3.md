ID: ISSUE-LOCAL-01M3NC14GE995V9E6F7GTYSQJ3
Title: record checkers stop at the first invalid input and hide the rest
Row: GATE-ISSUE-INDEX-TABLE-SHAPE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-28
Updated: 2026-09-28
Closed: -

## Problem

Both record validators abort on the first invalid input, so one run reveals only one defect and each fix exposes the next only after a new run (this masked three separate defects behind each other in the ORPHAN-MODEL-ROWS repair). scripts/agent-issue-index.py load_local_files raises IssueRecordError at the first file whose record fails validate_issue_record, so ten broken issue files report one error and --check shows only that one. scripts/check-agent-record.py parse_claim_rows drops any row whose cell count or state cell is malformed (bare continue), which both corrupts the matrix counts AND silently removes the row from every downstream contract check (owner/claim, anchors, spec) - a row with one missing cell reports a pipe-count error and nothing else, exactly the MODEL-DSV41 shape.

## Resolution

-
