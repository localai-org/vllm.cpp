ID: ISSUE-LOCAL-01M3JJ9CE2WH0KE640HQFGACNH
Title: No .gitattributes line-ending policy: agent-issue-index.py --refresh can never pass on a CRLF checkout
Row: BACKEND-VULKAN
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

The repository has NO line-ending policy. .gitattributes exists but contains only a linguist-generated rule for the vendored Triton AOT artifacts, so on any contributor machine with core.autocrlf=true (the default on Windows) every text file is checked out CRLF.

Several tools compare file content as EXACT BYTES, and one of them is now structurally unable to pass on such a checkout. scripts/issue_records.py _archive_evidence_matches_source splits the frozen archive on b newline and requires the raw bytes at a declared line to equal the evidence quoted in an _intake issue. parse_issue_file normalises the quoted evidence through normalize_body, which runs _normal_newlines and strips the carriage return. So on a CRLF checkout the archive line ends CR and the quote does not, and they can never be equal.

MEASURED 2026-09-27 on a Windows checkout at 1de097c46, for ISSUE-GH-148: the archive line and the quoted evidence are byte-identical except for one trailing CR, EXACT MATCH False, match-ignoring-CR True. The committed blob is pure LF (0 CR, 886 LF in .agents/completed/issue-index.md), so the repository content is correct and this is purely a checkout artifact. Consequence: scripts/agent-issue-index.py --refresh fails permanently for Windows contributors, which silently disables the canonical issue index, and the failure reads like a record defect rather than an environment one.

The fix is the line endings, NOT the checker. Making the comparison CR-tolerant would widen a checker to make a gate green, which AGENTS.md forbids without a spec and red-before evidence, and byte-exactness is the invariant the record actually depends on.

SCOPE IS DELIBERATELY NARROW, and that is an evidence-backed choice rather than caution. A blanket * text=auto eol=lf was considered and rejected: scanning all 7114 tracked text blobs finds exactly 3 containing CRLF, and all three are captured bench-evidence logs (docs/bench-evidence/oracle-vllm-gfx1151-20260903/job-phase2.txt, docs/bench-evidence/strix-kernel-trace-3015-20260907/profile-dependencies.log.txt, docs/bench-evidence/vllm-gguf-plugin-thor-20260903/gen-20260903T012806Z.log) where the carriage returns are part of the recorded artifact and must never be renormalised. Those are marked -text so a future blanket rule cannot silently mutate evidence.

The rules cover only the paths a byte-exact checker reads: the .agents markdown records and frozen archive, the scripts themselves (whose literals are compared), and the DeepSeek-V4-Vision fixture directory that check-deepseek-v4-vision-manifests.py byte-compares against generated LF-terminated JSON. ab-arms-differ.py and check-release-binary-contract.py were audited and are NOT line-ending sensitive (one scans an ELF for a byte root, the other splits on a NUL pair), so they are left alone.

## Resolution

-
