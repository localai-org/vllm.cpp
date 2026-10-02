ID: ISSUE-LOCAL-01M3S6GT9BQR7H0T3M2HRXR0A1
Title: test_q4exp_layerfp_diff fails on main: the publisher no longer finds its docs/USAGE.md anchor
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

tests/scripts/test_q4exp_layerfp_diff.py fails 2 of 38 cases at base 46f7c27ca (and with the cohere2-moe change): MetricSpread.test_the_PUBLISHER_reproduces_docs_USAGE_md ('docs/USAGE.md: the publisher cannot find where this surface states the under-report median and p05..p95') and test_the_PUBLISHER_is_NOT_VACUOUS_on_a_moved_digit_or_a_lost_anchor ('the site under test no longer matches: the four no-change shares'). scripts/agent-preflight.sh --staged reports it. Found during the Cohere2Moe port; unrelated to it and not fixed there.

## Resolution

-
