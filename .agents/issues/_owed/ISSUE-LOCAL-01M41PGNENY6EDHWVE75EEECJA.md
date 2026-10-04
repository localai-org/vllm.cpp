ID: ISSUE-LOCAL-01M41PGNENY6EDHWVE75EEECJA
Title: pr-size base lane lacks pwsh — red-before evidence cannot reproduce
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-03
Updated: 2026-10-03
Closed: -

## Problem

scripts/check-windows-portability.py skips its PowerShell AST stage when pwsh is absent (shutil.which('pwsh') is None at scripts/check-windows-portability.py:2029). The pr-size job (.github/workflows/ci.yml, job pr-size) runs check-pr-size.py's semantic-evidence gate on ubuntu-latest without installing pwsh, so the BASE re-run of tests/scripts/test_check_windows_portability.py silently skips the ps1 stage and stays green; check-pr-size.py:756 then rejects the changed test with 'BASE checker stayed green; changed test is not semantic evidence'. The same test reproduces RED on any lane that has pwsh. The job lanes are environment-asymmetric. Fix: install pwsh in the pr-size job so HEAD and BASE evidence lanes match the lanes where the test is red.

## Resolution

-
