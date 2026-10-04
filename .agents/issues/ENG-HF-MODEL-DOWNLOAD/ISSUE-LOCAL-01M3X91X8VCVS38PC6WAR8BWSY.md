ID: ISSUE-LOCAL-01M3X91X8VCVS38PC6WAR8BWSY
Title: Quickstart still declares repaired Hub redirects blocked
Row: ENG-HF-MODEL-DOWNLOAD
State: OPEN
Kind: documentation
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-02
Updated: 2026-10-02
Closed: -

## Problem

docs/QUICKSTART.md says repository downloads are unavailable because of relative redirects. Both downloader loops now use HfResolveUrl and retain regression tests. Remove the stale blocker without promoting the mounted-container evidence to a repository-download gate.

## Resolution

2026-10-02: corrected in 6441747d7; focused CPU documentation checks and independent review pass. Awaiting upstream PR integration. See .agents/specs/docs-quickstart-hub-refresh.md.
