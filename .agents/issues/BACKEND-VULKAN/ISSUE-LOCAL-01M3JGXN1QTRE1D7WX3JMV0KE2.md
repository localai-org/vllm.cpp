ID: ISSUE-LOCAL-01M3JGXN1QTRE1D7WX3JMV0KE2
Title: There is no runtime device selector: VLLM_CPP_DEVICE is read nowhere, so every non-CUDA gate selects its device by accident
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

The engine picks its device in src/vllm/entrypoints/model_loader.cpp via CurrentPlatform(), which walks {kCUDA, kXPU, kVULKAN, kMETAL, kCPU} and takes the FIRST backend that probed a device. Nothing overrides that. The env var the campaign recorded, VLLM_CPP_DEVICE, is read NOWHERE: grep over the whole tree returns only record files, never src/, include/, tests/ or examples/.

So `VLLM_CPP_DEVICE=vulkan test_opt_paged_engine` selected Vulkan only because those builds had no CUDA compiler. benchmark-record.md already MEASURED both ways on one source: with /usr/local/cuda/bin on PATH at configure time the identical command reports 'the engine selected device type 1' (kCUDA) and passes 6/6; without it, 'device type 3' (kVULKAN), 6/6, 0 declines.

Why this is filed now rather than read and skipped: the estate gained a SECOND accelerator host (the Intel test host, Intel Arc Pro B60, no CUDA toolchain at all). A gate that selects its device by build accident is exactly the failure that a new box invites -- on a host with no CUDA the placebo is indistinguishable from a working selector, so the first Vulkan-only venue cannot tell the difference. The test already prints the truth ('the engine selected device type N') and load-direct-upload.md already uses the correct idiom, so the defect is that no selector EXISTS, not that the evidence is unobtainable.

Scope: add a real runtime device override read at the SelectQueue seam, make it fail LOUDLY on an unknown value rather than silently falling through, and pin the selection in the test's own assertions. Then replace the remaining VLLM_CPP_DEVICE command citations in the record with the BACKEND PROOF form. Not attempted in the records-only VK-I change, because it is a code change to the device seam and needs its own row, spec and gate.

## Resolution

-
