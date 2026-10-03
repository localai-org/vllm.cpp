ID: ISSUE-LOCAL-01M3Z01WBRQ9NEBDCGPH6CMM8V
Title: q4exp USAGE publisher tests red on arrival — the USAGE sites never landed
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-02
Updated: 2026-10-02
Closed: -

## Problem

tests/scripts/test_q4exp_layerfp_diff.py::_USAGE_SITES defines five figure sites that docs/USAGE.md has never carried in any commit; MetricSpread::test_the_PUBLISHER_reproduces_docs_USAGE_md and test_the_PUBLISHER_is_NOT_VACUOUS_on_a_moved_digit_or_a_lost_anchor fail red on arrival. The figures are the seeded 400-draw MetricSpread control, already published verbatim in the scripts/q4exp-layerfp-diff.py docstring and runtime stdout; only the USAGE surface never landed. Repair: write the Qwen4-EXP LayerFp diagnostics section in docs/USAGE.md sourced from that committed control.

## Resolution

-
