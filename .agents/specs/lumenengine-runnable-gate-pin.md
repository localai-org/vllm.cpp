# LumenEngine runnable gate baseline

## Scope

Issue: ISSUE-LOCAL-01M641PXTYNV999F8RZAXJDSE9.
Row: ENG-LUMENENGINE-BRAND.
Pinned base: 9f4b15d93.
Repair the exact runnable baseline record. Do not change classifier semantics or tests.

## Current evidence

The derived runnable set contains 53 IDs and the baseline contains 52.
Only ENG-LUMENENGINE-BRAND is missing. No baseline ID left the derived set.
The three Pascal and Volta IDs already appear in commit ea8299d89.
The existing branding spec's Gates section invokes scripts/agent-preflight.sh --quiet.
This is a runnable command that can fail. The exact-set suite reports 15 failures.

## Design and source

Add only ENG-LUMENENGINE-BRAND to scripts/check-gate-commands.py RUNNABLE_BASELINE.
Add its dated reason beside the entry. Preserve every existing entry byte-for-byte.
The source contract is scripts/check-gate-commands.py lines 14 to 21 and its baseline commentary.
No vLLM runtime behavior or oracle measurement changes.

## Gates

Run python3 tests/scripts/test_check_gate_commands.py before the repair and capture failure.
Run the same 61 tests after the repair, then python3 scripts/check-gate-commands.py --check.
Delete the added entry in scratch and require the exact-set test to fail.
Restore the source byte-for-byte and repeat the exact-set test.
Check unchanged entries, classifier bytes, canonical records, whitespace and commit style.
A fresh independent reviewer and the operator rerun remain required.

## Risks and stop conditions

A broader set change would need separate ownership evidence.
Stop if the derived addition is not exactly ENG-LUMENENGINE-BRAND.
Do not include unmerged FP16 work, alter lifecycle records or rewrite unrelated specs.
No GPU work or remote writes are authorized by this repair.

## Now

ACTIVE. Issue and spec precede the baseline edit. One PR follows recorded integration policy.

## Owed

Independent review, operator verification and landing remain required.

## Implementation evidence

On 2026-10-29, the existing 61-test suite failed 15 tests before the change
and passed all 61 afterward. The command was
`python3 tests/scripts/test_check_gate_commands.py`.
`python3 scripts/check-gate-commands.py --check` passed.

Deleting the new ID made
`RatchetTests.test_the_baseline_matches_the_shipped_record` fail.
Restoring the source byte-for-byte made that test pass.
Removing the four added lines reproduces the complete script at 9f4b15d93
byte-for-byte. Existing entries and classifier semantics are unchanged.

`python3 scripts/check-agent-record.py` and `git diff --check` passed.
Logs reside in `/tmp/lumen-gate-baseline-evidence`.
The startup preflight ran for about 45 seconds before termination; it is
incomplete. No GPU work ran. The issue stays open until landing.
