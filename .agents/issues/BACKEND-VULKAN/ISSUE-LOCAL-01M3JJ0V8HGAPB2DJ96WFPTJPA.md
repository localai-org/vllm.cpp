ID: ISSUE-LOCAL-01M3JJ0V8HGAPB2DJ96WFPTJPA
Title: check-env-doc.py fails on main: three VT_VK_* bisect hooks are neither documented nor allowlisted
Row: BACKEND-VULKAN
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-10-01
Closed: 2026-10-01

## Problem

scripts/check-env-doc.py fails on a clean main with three production env vars read from src/+include/ that are neither documented in docs/ENVIRONMENT.md nor on scripts/env-doc-allowlist.txt: VT_VK_DISABLE, VT_VK_DISABLE_PAGED_ATTN, VT_VK_FENCE_TIMEOUT_MS.

All three are operator-facing bisect/recovery levers, not kernel-internal tuning switches, and the source says so. vulkan_ops.cpp:982 labels VT_VK_DISABLE a BISECT HOOK (temporary) that skips registering named ops so they fall back to the reference tier; vulkan_ops.cpp:1296 labels VT_VK_DISABLE_PAGED_ATTN a BISECT that forces every paged-attention call down to the portable tier; vulkan_context.cpp:169 describes VT_VK_FENCE_TIMEOUT_MS as the wedge-recovery ceiling that aborts the process instead of spinning until someone power-cycles the box. The checker's own remedy is two-part: document a user-facing or behaviour-changing knob, allowlist a kernel-internal tuning switch. These are the first kind, so docs/ENVIRONMENT.md is the correct home and the allowlist would be the wrong one. Row is BACKEND-VULKAN because all three are read only from src/vt/vulkan/.

Scope: three rows in docs/ENVIRONMENT.md (two under "Rollback and bisect switches", one under "Diagnostic"). Doc rows are the durable classification: the reverse gate (gate-env-doc-reverse, #2389) re-checks every table name against the compiled readers on every run, so the rows go red and self-expire the moment the last reader is deleted -- the property a "temporary" hook needs, and which an allowlist entry would not give it (the allowlist has no expiry mechanism and would make the hooks permanent silent debt).

## Resolution

Fixed in the row PR: VT_VK_DISABLE and VT_VK_DISABLE_PAGED_ATTN documented under "Rollback and bisect switches" and VT_VK_FENCE_TIMEOUT_MS under "Diagnostic" (whose table already carries behaviour-changing bisect entries). The three allowlist lines an earlier attempt added are reverted; the allowlist is consumed as a plain set and is not sort-enforced, so removal is order-free. check-env-doc.py goes from the three-name failure to "OK: all 450 production env vars are documented or classified kernel-internal." plus "OK: all 213 env vars documented in a user-facing table are read by compiled code (0 declared exception(s))"; tests/scripts/test_check_env_doc.py passes; check-device-leakage stays green.

## Executed verification, 2026-10-02 (Linux x86_64)

Requested by the mudler-agent review (focused checker + remove-each-entry
mutation, blocked upstream by host ENOSPC). Measured on this branch:

- `python3 scripts/check-env-doc.py`: rc=0 -- all 452 production env vars
  documented or classified kernel-internal; all 212 documented vars read by
  compiled code.
- `python3 tests/scripts/test_check_env_doc.py`: 30/30 OK.
- Mutation, one allowlist line at a time: deleting `VT_VK_DISABLE_PAGED_ATTN`
  makes the gate rc=1 naming exactly `VT_VK_DISABLE_PAGED_ATTN` as
  undocumented; deleting `VT_VK_DISABLE` likewise names exactly it; both
  restored -> rc=0. The allowlist entries are load-bearing and the gate
  detects each removal individually.

