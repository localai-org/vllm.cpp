ID: ISSUE-LOCAL-01M433BAC470XAV8VYFF8KJ1WD
Title: ci wiring: vulkan platform-gate log and pr-size base-lane residuals
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

Two CI-infra residuals on top of the pwsh install (3b812f596, eb815c226). (1) build-test-vulkan's Platform seam gate greps platform-gate.log for 'platform vulkan get_device_capability()' but its invocation ('./build-vulkan/tests/test_platform | tee platform-gate.log') does not capture the output surface the doctest MESSAGE() line is emitted on -- no 2>&1 and no --success, unlike the proven-good metal lane ('--no-colors=1 2>&1') -- so the step fails 'pattern not found' on a message the job demonstrably printed in an earlier step's output. (2) the pr-size semantic-evidence lanes stay asymmetric despite the installed pwsh: the lane's sanitized PATH is os.defpath (/bin:/usr/bin) plus the private tools directory (_sanitized_env, scripts/check-pr-size.py:881), pwsh is absent from EVIDENCE_REQUIRED_TOOLS (scripts/check-pr-size.py:852), and the install lands at /usr/local/bin/pwsh -- so shutil.which('pwsh') inside check-windows-portability.py:2028 misses it in BOTH lanes, the PowerShell AST stage silently skips, and the gate rejects the evidence with 'BASE checker stayed green ... changed test is not semantic evidence'. This also resolves ISSUE-LOCAL-01M41PGNENY6EDHWVE75EEECJA.

## Resolution

-
