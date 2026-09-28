ID: ISSUE-LOCAL-01M3JPFDVH2CWM5FV4NCSD268Z
Title: merge c12b376b2 deleted the record-anchor ratchet from check-agent-record.py, and the row, the baseline, --report and the test class all still claim it runs
Row: ENG-RECORD-ANCHOR-RATCHET
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

The record-anchor ratchet was deleted from scripts/check-agent-record.py by a merge resolution, and three gates went quiet about it.

c12b376b2 ("Merge branch 'row/KERNEL-GEMM-CPU-ELEM-A76' of https://github.com/richiejp/vllm.cpp into tmp-merge-361") has two parents. Parent 1, 4fcce96b5, is the main line and carries scripts/check-agent-record.py at 2446 lines with the whole anchor subsystem present. Parent 2, 858560fd4, is the feature branch, was branched before the ratchet landed, and carries the same file at 1215 lines with the subsystem absent. The parents are divergent (git merge-base --is-ancestor c12b376b2^2 c12b376b2^1 exits non-zero), so this was a real three-way merge, and the resolution took parent 2's version of this file. 1231 lines went with it.

Gone: BARE_CITATION_RE, cell_citations, classify_citation, RECORD_ANCHOR_STATES, check_record_anchors, RecordAnchorResult, write_record_anchor_baseline, load_record_anchor_baseline, record_anchor_report, scan_record_anchors, extract_links, strip_code_spans, link_bases, issue_references_in_text, discover_issue_references, branch_issue_references, canonical_intake_debt and check_canonical_issue_references. The subsystem was added by 678fc672c (feat(ENG-RECORD-ANCHOR-RATCHET), #632/#851), which took the file to 2040 lines, and reached 2446 before the merge.

Four things stayed true after the deletion and each is worse than a stale citation:

1. The row is still ACTIVE. .agents/engine-matrix.md:231 reads "parser + classifier + ratchet in check-agent-record.py" and names all five symbols. check-symbol-anchors is red on exactly those five, and it is the only gate that noticed.

2. The baseline is orphaned. scripts/record-anchor-baseline.json is read by nothing under scripts/. The only remaining reference in the tree is tests/scripts/test_agent_record.py:1856, which writes a synthetic copy into a temp dir. The ratchet had a two-way floor: it refused to ratchet upward. There is no longer anything to hold the floor.

3. --report is now a no-op, and two callers still pass it. scripts/check-agent-record.py main() takes no arguments, so the flag is silently ignored and the checker exits 0. .github/workflows/ci.yml:187 runs `python3 scripts/check-agent-record.py --report` and scripts/agent-preflight.sh:418 documents that CI "prints it". Neither prints anything now. A gate that was wired to a flag the gate no longer reads looks wired.

4. Its own test class is failing. tests/scripts/test_agent_record.py is 2651 lines and runs 106 tests with 26 failures and 32 errors. RecordAnchorRatchet (line 1754) calls agent_record.RECORD_ANCHOR_STATES, agent_record.RecordAnchorResult and agent_record.write_record_anchor_baseline, none of which exist. The file also carries a line-floor test demanding at least 2446 lines, which is parent 1's count, so the suite encodes the pre-merge size.

The restoration is NOT attempted here. Reinstating 1231 lines reverses a merge that is on main, and the repo's own precedent for undoing record-rot at this size is f39808373 ("preserve retired gate evidence", +823 lines) which came in with a spec and a fresh review. This issue files the finding with the exact commits so that decision is taken deliberately rather than inherited.

What this change does: correct the cell to state the subsystem is absent, and record the above. What it does not do: reinstate code, change the row's state, or touch the checker.

## Resolution

-
