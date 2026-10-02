ID: ISSUE-LOCAL-01M3JJ9CE2WH0KE640HQFGACNH
Title: No .gitattributes line-ending policy: agent-issue-index.py --refresh can never pass on a CRLF checkout
Row: BACKEND-VULKAN
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-10-02
Closed: -

## Problem

The repository has NO line-ending policy. .gitattributes exists but contains only a linguist-generated rule for the vendored Triton AOT artifacts, so on any contributor machine with core.autocrlf=true (the default on Windows) every text file is checked out CRLF.

Several tools compare file content as EXACT BYTES, and one of them is now structurally unable to pass on such a checkout. scripts/issue_records.py _archive_evidence_matches_source splits the frozen archive on b newline and requires the raw bytes at a declared line to equal the evidence quoted in an _intake issue. parse_issue_file normalises the quoted evidence through normalize_body, which runs _normal_newlines and strips the carriage return. So on a CRLF checkout the archive line ends CR and the quote does not, and they can never be equal.

MEASURED 2026-09-27 on a Windows checkout at 1de097c46, for ISSUE-GH-148: the archive line and the quoted evidence are byte-identical except for one trailing CR, EXACT MATCH False, match-ignoring-CR True. The committed blob is pure LF (0 CR, 886 LF in .agents/completed/issue-index.md), so the repository content is correct and this is purely a checkout artifact. Consequence: scripts/agent-issue-index.py --refresh fails permanently for Windows contributors, which silently disables the canonical issue index, and the failure reads like a record defect rather than an environment one.

The fix is the line endings, NOT the checker. Making the comparison CR-tolerant would widen a checker to make a gate green, which AGENTS.md forbids without a spec and red-before evidence, and byte-exactness is the invariant the record actually depends on.

SCOPE REVISED 2026-10-02 after review asked for the checkout test. The first cut covered only .agents/**/*.md, scripts/*.py and the DeepSeek-V4-Vision fixture dir, reasoning that those were the paths a byte-exact checker reads. A core.autocrlf=true clone + checkout of the branch showed that framing was wrong in BOTH directions:

- The record surface is wider than *.md. On the autocrlf checkout the .md rows stayed LF, but .agents/completed/*.csv manifests, .agents/evidence/** JSON/stdout captures, .agents/specs/*.log and *.patch, and .agents/scripts/*.sh / *.py all arrived CRLF -- the same byte-exact hazard one directory over.
- The policy normalises nothing that exists. Scanning the index finds ZERO text blobs outside docs/bench-evidence containing \r\n, and ZERO binary blobs under .agents/ (2488 files) or scripts/ (313), so a whole-tree * text=auto eol=lf is safe and rewrites no stored content -- it only stops the class from arriving.

The policy is therefore `* text=auto eol=lf` with two byte-preserving pins. docs/bench-evidence/** is pinned -text directory-wide, not file-by-file: the directory carries three true-CRLF text logs (job-phase2.txt, profile-dependencies.log.txt, gen-20260903T012806Z.log) PLUS eight more text logs with lone progress-bar CRs (limb3-strict-gate gen-*.log, job-phase3.txt, q4km-neartie harness-stdout-*.txt), and any of them committed from an autocrlf checkout would be rewritten. tests/parity/goldens/** is pinned -text because the .npy/.i32/.raw fixtures and generated manifests are bytes, not lines. After the change, the same autocrlf clone produces zero CR-bearing files outside the pinned evidence directory.

## Resolution

-
