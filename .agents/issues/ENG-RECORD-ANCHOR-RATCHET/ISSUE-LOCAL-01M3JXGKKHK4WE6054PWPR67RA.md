ID: ISSUE-LOCAL-01M3JXGKKHK4WE6054PWPR67RA
Title: Merge c12b376b2 deleted the record-anchor subsystem; its suite survived and 58 of 106 cases have been red since, and the data file it reads outlived the code
Row: ENG-RECORD-ANCHOR-RATCHET
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-10-02
Closed: 2026-10-02

## Problem

scripts/check-agent-record.py lost 23 definitions to merge c12b376b2 (Merge branch row/KERNEL-GEMM-CPU-ELEM-A76 ... into tmp-merge-361, divergent parents 858560fd4 feature-side and 4fcce96b5 main-side; the resolution took the feature side, 1215 lines against 2446). All 23 are absent from the whole scripts/ tree, including the 18 names .agents/specs/record-anchor-ratchet.md describes and scripts/record-anchor-baseline.json still names in its own _comment. The suite did NOT go with them: tests/scripts/test_agent_record.py runs 106 cases and 58 fail today (26 failures, 32 errors), naming extract_links (13 cases), scan_record_anchors (8), link_bases (3), strip_code_spans, RecordAnchorResult, RECORD_ANCHOR_BASELINE, and row_ids cascading off that dataclass. The failing set spans three subsystems, not one: the anchor ratchet, the markdown fence and inline-span link extractor, and the claim-state checks. So the tree contradicts itself today: its own tests demand an implementation the same commit removed, and scripts/record-anchor-baseline.json survives as a 898-byte file read by nothing. Meanwhile check-agent-record.py is rc=0 on main (agent record OK: ENGINE=179 MODEL=384 QUANT=87 KERNEL=60 BACKEND=90), which is GREEN BECAUSE THE RATCHET IS ABSENT rather than because the records are clean. The spec's ## Now still reads ACTIVE -- the parser, the classifier, the baseline and the cases are implemented and green -- which is false of this tree, and it is the same ## Now-contradicts-the-code class already found in mimov2.md, in five of nine missing-arch specs, and in gliner2.5-decide.md. MEASURED CONSEQUENCE OF A VERBATIM TRANSPLANT, taken by checking 4fcce96b5 out over the current tree in a scratch worktree: it parses, and it goes red. RECORD ANCHOR REGRESSION in bucket stale: 39 > baseline 0, and in bucket broken: 5 > baseline 0, so about 44 citations have rotted since the subsystem was removed. A straight transplant also produces two further reds that are artefacts of resurrecting code HEAD has since deliberately moved on from: 50 roadmap_v1.md issue-table errors, because HEAD replaced the old check_issue_records with check_issue_table and the old checker also carried link_bases two-base archive behaviour that HEAD flattened to link_base; and the 9 _intake plus 3 orphan-row records that canonical_intake_debt reports, which are exactly the residues #3333 and #3342 fix. So the restoration is bounded only if the ratchet is restored unwired: the tests call scan_record_anchors, check_record_anchors and main(['--write-baseline']) DIRECTLY rather than through the default run path, which is verified in the suite at test_new_rot_fails_the_gate:1898 and test_a_baseline_is_never_banked_from_a_tree_with_record_errors:1905, so an unwired main() satisfies every one of the 58. HEAD main() at :1205 takes no arguments and its --report, which .github/workflows/ci.yml:187 and scripts/agent-preflight.sh:418 both pass, is silently ignored.

## Resolution

upstream/main now carries the full subsystem WIRED, which supersedes this
row's restored-but-unwired transplant: scan_record_anchors at
check-agent-record.py:1632, check_record_anchors at :1728 called from
main() inside the errors gate, --report prints the offenders, and
--write-baseline writes scripts/record-anchor-baseline.json only after
the gate passes. check_claim_state_consistency is called at :1980 and
canonical_intake_debt / check_canonical_issue_references at :2305. The
baseline file exists upstream. On the merged tree the checker's
remaining ERRORs are the shared issue-reference debt identical on
upstream/main, not this row's. This record is what the pull request
still adds.
