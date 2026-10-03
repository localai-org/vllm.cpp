ID: ISSUE-LOCAL-01M3WZNRGJPME7P88GPPBW57GA
Title: Remove the temporary VT_VK_DISABLE and VT_VK_DISABLE_PAGED_ATTN bisect hooks once the Vulkan op investigation closes
Row: BACKEND-VULKAN
State: OPEN
Kind: tech-debt
GitHub: 3372
Mirror: SYNCED
Availability: FULL
Created: 2026-10-01
Updated: 2026-10-02
Closed: -

## Problem

src/vt/vulkan/vulkan_ops.cpp carries two temporary bisect hooks from the Vulkan op investigation: VT_VK_DISABLE (vulkan_ops.cpp:982, source-labelled BISECT HOOK (temporary)) skips registering the named ops so they fall back to the reference tier, and VT_VK_DISABLE_PAGED_ATTN (vulkan_ops.cpp:1296, source-labelled BISECT) forces every paged-attention call down to the portable tier through the provider fallback. Both are on scripts/env-doc-allowlist.txt, so check-env-doc.py accepts them without a doc row, and docs/ENVIRONMENT.md has no rows for them. The allowlist has no expiry mechanism: nothing turns red when the hooks stay, which is why this record owns the removal. Removal means deleting the two hooks, their getenv readers and the two allowlist entries together in one change. Out of scope: VT_VK_FENCE_TIMEOUT_MS (src/vt/vulkan/vulkan_context.cpp:169) is a wedge-recovery guard, not a bisect hook, and stays. Removal trigger: the Vulkan op investigation closes (the MoE router/combine and paged-attention paths confirmed correct on the Vulkan tier); then delete both hooks, their readers, and the two allowlist entries in one change.

## Resolution

-
