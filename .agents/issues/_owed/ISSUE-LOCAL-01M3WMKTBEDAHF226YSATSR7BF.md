ID: ISSUE-LOCAL-01M3WMKTBEDAHF226YSATSR7BF
Title: five preflight doc gates red on main
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-01
Updated: 2026-10-01
Closed: -

## Problem

On origin/main the preflight carries five red record/doc gates: check-release-binary-contract and check-windows-release-state (roadmap_v1.md lost its release-lifecycle anchors and the ENG-RELEASE-WINDOWS state anchor in the record restructure), check-benchmark-index (16 detail files in docs/benchmarks/ are orphans of the BENCHMARKS.md index), check-env-doc (VT_VK_DISABLE, VT_VK_DISABLE_PAGED_ATTN, VT_VK_FENCE_TIMEOUT_MS read from src/ are neither documented nor allowlisted), and check-oracle-pins (the 2026-09-26 sync 4f11dfc10 advanced the parity pin to a7c23ac96d/0.3.0.dev267 but left the restated spans in NOW.md, FEATURES.md, how-we-measure.md, speculative-decoding.md and vllm-online-serving.md at the prior e126687a9a/0.28.1rc1.dev132). Repair the records to the checker contracts without weakening any checker.

## Resolution

Repaired on row/preflight-gate-repairs: (1)+(2) roadmap_v1.md release-lifecycle and ENG-RELEASE-WINDOWS anchors restored from 10a64f827/a0a8fedbe; (3) all 16 docs/benchmarks detail files indexed in BENCHMARKS.md; (4) VT_VK_DISABLE and VT_VK_DISABLE_PAGED_ATTN allowlisted as kernel-internal bisect hooks, VT_VK_FENCE_TIMEOUT_MS documented in docs/ENVIRONMENT.md; (5) parity pin restated to a7c23ac96d/0.3.0.dev267 in .agents/oracles/vllm.md and the five prose surfaces the 2026-09-26 sync 4f11dfc10 left behind. All five checkers exit 0; check-agent-record and the record suites stay green.
