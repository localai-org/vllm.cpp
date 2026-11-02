# Legacy CUDA gate record repair

Issue: `ISSUE-LOCAL-01M63NAHP61SNGD5ZBQM9BGCA8`.
Base: `03b19115a83f1c257b630c1e3bece7361ce08e95`.
Spec-first record commit: `a7900dc9b`.

## Observed gaps

All three ACTIVE matrix rows resolve the shared spec. Its Gates section names
`scripts/agent-preflight.sh --quiet`, so all three enter the runnable inventory.
The frozen implementation omits them from the exact baseline. The unchanged
61-case gate-command suite fails 15 assertions naming SM060, SM061, and SM070.

The live audit also lacks exact SM070 campaign provenance on the frozen head.
Its original preflight reports `BACKEND-CUDA-SM070` as abandoned.
The same shipped-record test fails on a detached frozen-head scratch checkout
and names only that row. It passes after the provenance commit. The record
commit names each covered row and the shared sm_60, sm_61, and sm_70 build work.
P100 and Volta runtime, real-checkpoint correctness, and performance stay pending.

## Repair and verification

Only the expected runnable inventory changes. The classifier, exact-set
assertion, live-audit rules, production code, and lifecycle states stay unchanged.
No CI registration is missing: preflight and CI already invoke both checkers.

- `python3 tests/scripts/test_check_gate_commands.py`: 61 tests pass.
- `python3 tests/scripts/test_audit_live_rows.py`: 58 tests pass.
- `python3 scripts/check-gate-commands.py --check`: exit 0.
- `python3 scripts/check-agent-record.py`: exit 0.
- `git diff --check`: exit 0.

Delete each added inventory membership independently. The existing shipped-set
assertion fails and names the deleted row. Restore the script byte-for-byte,
and the assertion passes. Each mutation uses a separate Python bytecode cache
to avoid equal-size, same-second source-cache reuse.

Logs live beside this report. Full repository preflight remains non-green
because the campaign separately records its existing environment failures.
The helper did not start another full preflight. The issue remains open until
its change lands. Independent review and operator verification remain required.
