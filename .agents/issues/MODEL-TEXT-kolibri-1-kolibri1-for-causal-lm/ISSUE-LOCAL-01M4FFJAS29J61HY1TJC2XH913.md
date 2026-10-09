ID: ISSUE-LOCAL-01M4FFJAS29J61HY1TJC2XH913
Title: NEON paged-attn TU breaks -Werror builds on non-aarch64 (unused Q()/Contig())
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: -

## Problem

PR #3425's new tests/vt/test_ops_paged_attn_neon.cpp defines anonymous-namespace helpers Q() and Contig() outside the __aarch64__ guard; the test body using them is inside the guard's #else branch, so on non-aarch64 (CI build-test-cpu, build-newest-gcc, both -Wall -Wextra -Werror) the helpers are defined-but-unused and the build fails. Found via main CI red after the branch's local aarch64 builds passed (helpers ARE used on aarch64). Fix: [[maybe_unused]] on both helpers.

## Resolution

-
