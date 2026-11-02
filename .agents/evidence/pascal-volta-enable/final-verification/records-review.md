# Scoped legacy CUDA record review

Head: `3c062a6273bacc057fdacf76d1763e808a3adce7`.
Base: `03b19115a83f1c257b630c1e3bece7361ce08e95`.
Scope: spec-first commit `a7900dc9b`, six-line runnable inventory repair, local issue, and evidence.
Scratch: `/tmp/lumen-records-review`, independent helper claim.

Findings: NONE.
Verdict: **PASS**.

## Static review

The only script change adds three exact expected entries and an explanatory comment. The classifier, assertions, tests, CI registration, and product code are unchanged. All three ACTIVE rows resolve the committed shared spec, whose Gates section names a runnable preflight command. The exact expected population therefore requires SM060, SM061, and SM070.

Both authored commit bodies name all three stable row IDs. This is truthful provenance for the shared tri-architecture build campaign, not a claim of unavailable P100 or Volta runtime results. The spec explicitly retains hardware and published-model gates as pending. Spec commit a7900dc9b precedes repair commit 3c062a627. The row-owned issue remains OPEN until landing and records dated resolution evidence.

## Independent verification

- `python3 tests/scripts/test_check_gate_commands.py`: PASS, 61 tests.
- Discovery of the same test module: PASS, 70 tests.
- `python3 -m unittest discover -s tests/scripts -p test_audit_live_rows.py`: PASS, 58 tests.
- `python3 scripts/check-gate-commands.py --check`: PASS, exit 0.
- `python3 scripts/check-agent-record.py`: PASS, exit 0.
- `git diff --check 03b19115a HEAD`: PASS, exit 0.

Logs: `/tmp/lumen-records-review-gates-final.log`, `/tmp/lumen-records-review-gates.log`, `/tmp/lumen-records-review-audit.log`, `/tmp/lumen-records-check-gates.log`, `/tmp/lumen-records-check-record.log`.

## Independent mutations

Deleted each added baseline row independently in the reviewer scratch script. Ran the unchanged `RatchetTests.test_the_baseline_matches_the_shipped_record` test with a separate Python bytecode cache for each mutation. Every deletion exits 1 and names exactly its missing row: SM060, SM061, and SM070. Restored the script byte-for-byte after each mutation. The final restored direct 61-test suite passes. Logs: `/tmp/lumen-records-BACKEND-CUDA-SM060-mutation.log`, `/tmp/lumen-records-BACKEND-CUDA-SM061-mutation.log`, `/tmp/lumen-records-BACKEND-CUDA-SM070-mutation.log`.

The reviewed worktree, branch, index, and head were not changed. Reviewer scratch `git status --short` is empty. No full repository preflight was run, per the coordinator's explicit scoped instruction.

## Remaining concern

This scoped record repair does not discharge the product head-dimension-256 Pascal prefill finding, the pending Release CUDA gate, outstanding mutation coverage, published-model gates, or unavailable hardware validation. The overall CUDA implementation remains FAIL pending separate repair and review.
