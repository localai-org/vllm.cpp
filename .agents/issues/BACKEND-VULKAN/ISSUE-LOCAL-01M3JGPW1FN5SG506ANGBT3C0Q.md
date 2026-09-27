ID: ISSUE-LOCAL-01M3JGPW1FN5SG506ANGBT3C0Q
Title: VK-I: B60 ReBAR retires the discrete staging path; the 'B60 is integrated' comment and the 'no Intel GPU' registry line are both false
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

Three records about the Intel Arc Pro B60 are wrong, and one planned deliverable is answered by hardware rather than by code.

(1) FALSE CODE COMMENT. src/vt/vulkan/vulkan_ops.cpp:2109 asserts 'The B60 is integrated'. Measured on garlic-clove with vulkaninfo, the device reports deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, with two separate heaps (20.91 GiB DEVICE_LOCAL, 23.44 GiB host-visible). The CODE is right for the stated WRONG reason: VulkanBackend::DeviceMemoryIsHostAddressable() (vulkan_backend.cpp:135) returns true unconditionally, and vulkan_context.cpp:873 prefers a DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type, which memoryTypes[3] and memoryTypes[6] (propertyFlags 0x0007) satisfy. The property the code depends on holds; the stated reason does not. This is a trap rather than a typo: a non-ReBAR Arc card would be discrete AND non-host-visible, and this comment would tell the next reader it is fine.

(2) FALSE REGISTRY LINE. .agents/environment.md asserts 'No Intel GPU exists on any box here, so BACKEND-XPU end-to-end work is HW-BLOCKED'. garlic-clove is an Intel Arc Pro B60 (8086:e211, ASUS subsys 1849:6023) on the xe driver.

(3) VK-I's STAGING-PATH DELIVERABLE IS ANSWERED, NOT BUILDABLE. vulkan-full-support.md scopes VK-I as 'The staging path for non-host-visible memory, and the gate re-run where Vulkan actually matters', and records the 2026-08-06 decision 'GB10 first, acquire later'. The acquire has happened, but the first half needs no code: ReBAR maps the B60's 20.91 GiB as host-visible, so the non-host-visible condition VK-I was written to handle does not arise on this card. Recording that as a measurement retires the risk; writing a staging path against it would be code for a condition this hardware does not have.

Also stale, and the reason this was not caught: the BACKEND-VULKAN matrix cell still describes the 2026-07-22 skeleton (8 native ops, 'no model runs on Vulkan'). The tree now registers 35 Vulkan ops including the full GDN/SSM set, GGUF keep-quant/TQ1_0, EXL3 and MoE, and vulkan_ops.cpp:2109 itself names this board.

## Resolution

-
