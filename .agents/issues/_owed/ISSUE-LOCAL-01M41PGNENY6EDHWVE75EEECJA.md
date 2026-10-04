ID: ISSUE-LOCAL-01M41PGNENY6EDHWVE75EEECJA
Title: pr-size base lane lacks pwsh — red-before evidence cannot reproduce
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-03
Updated: 2026-10-04
Closed: -

## Problem

scripts/check-windows-portability.py skips its PowerShell AST stage when pwsh is absent (shutil.which('pwsh') is None at scripts/check-windows-portability.py:2029). The pr-size job (.github/workflows/ci.yml, job pr-size) runs check-pr-size.py's semantic-evidence gate on ubuntu-latest without installing pwsh, so the BASE re-run of tests/scripts/test_check_windows_portability.py silently skips the ps1 stage and stays green; check-pr-size.py:756 then rejects the changed test with 'BASE checker stayed green; changed test is not semantic evidence'. The same test reproduces RED on any lane that has pwsh. The job lanes are environment-asymmetric. Fix: install pwsh in the pr-size job so HEAD and BASE evidence lanes match the lanes where the test is red.

## Resolution

RESOLVED 2026-10-04, row/ci-wiring-residuals. The pwsh install at /usr/local/bin (3b812f596) was invisible to both evidence lanes: _sanitized_env PATH is os.defpath (/bin:/usr/bin) plus the private tools directory (scripts/check-pr-size.py:881-892), and pwsh was absent from EVIDENCE_REQUIRED_TOOLS (scripts/check-pr-size.py:852). Declaring pwsh for tests.scripts.test_check_windows_portability puts the same pwsh on both lanes' PATH, so the PowerShell AST stage runs at HEAD and at BASE and red-before reproduces. Superseded-in-part by ISSUE-LOCAL-01M433BAC470XAV8VYFF8KJ1WD for the record of the vulkan half.
