ID: ISSUE-LOCAL-01M3V01R90QQTQ08BWSBM7AGN2
Title: kCausalConv1dFwd (GDN prefill conv) has no native Vulkan kernel
Row: BACKEND-VULKAN
State: CLOSED
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-01
Updated: 2026-10-01
Closed: 2026-10-01

## Problem

vt::CausalConv1dFwd, the GDN PREFILL depthwise causal conv, is not registered for DeviceType::kVULKAN, so every GDN prefill on a Vulkan queue runs it on the portable CPU reference tier, once per GDN layer, behind a full batch drain. src/vt/vulkan/vulkan_ops.cpp left it as a follow-up because the op reads the OLD conv-state window and overwrites it in the same call; the comment names the two safe dispatch shapes (a serial invocation per (sequence, channel) over the whole token range, or a buffered old row). .agents/specs/vulkan-full-support.md section 6.0a lists it as one of the two remaining reference-tier declines.

## Resolution

Closed by row/BACKEND-VULKAN-CONVFWD: native vt_causal_conv1d_fwd, default one invocation per (sequence, channel) serial over its tokens with the old window copied before the write-back; opt-in token split (VT_VULKAN_CONV_TARGET_GROUPS) giving every t < width to block 0; unservable shapes decline through GetOpFallback; VT_VULKAN_CONV_FWD=0 keeps the reference tier. Gated by tests/vt/test_vulkan_backend.cpp against the CPU oracle; the split arm is red on llvmpipe with the pre-fix plain split and green with this PR.
