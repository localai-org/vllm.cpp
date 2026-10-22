ID: ISSUE-LOCAL-01M5H2VRQ270C05ZHSA6V8EQDA
Title: Re-capture the Kolibri-1 llama.cpp per-position rows at the pinned release b11765
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-22
Updated: 2026-10-22
Closed: -

## Problem

The scoped llama.cpp oracle for Kolibri-1 (.agents/oracles/llama-cpp-kolibri1.md) is pinned at release b11765 (9df0b73c), which carries upstream #30865, the removal of the redundant F32 precision overrides in the Kolibri-1 router. The committed per-position rows in tests/vllm/models/kolibri1_llamacpp_topk_rows.json were captured at b11709 (62318cd57), before that change, so they do not describe the pinned oracle's router arithmetic. Re-capture the rows at b11765, record the delta against the b11709 capture, and only then compare any NEW llama.cpp number against them.

## Resolution

-
