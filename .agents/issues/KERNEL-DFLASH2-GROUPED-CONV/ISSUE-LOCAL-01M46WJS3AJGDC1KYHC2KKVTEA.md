ID: ISSUE-LOCAL-01M46WJS3AJGDC1KYHC2KKVTEA
Title: rocm: DFlashGroupedConv has no ROCm kernel; test_dflash2_exl3_reach fails on gfx1100
Row: KERNEL-DFLASH2-GROUPED-CONV
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-05
Updated: 2026-10-05
Closed: -

## Problem

test_dflash2_exl3_reach fails on gfx1100 at branch rocm-gfx11-exl3-perf base e6e492238 AND at 8cc6aa921 (pre-existing, not introduced by the frac-rate change): 'no kernel for op DFlashGroupedConv (id 25) on device rocm (type 5), and the portable CPU reference tier is NOT eligible ... unified memory is false'. DFlashGroupedConv is registered for CUDA and CPU only (cuda_ops.cu:4177, cpu_ops.cpp:4477); nothing registers it for ROCm. The test's model+draft engine selects the ROCm device and throws during generate. Reproduce: ctest -R test_dflash2_exl3_reach in vllmcpp:git-ae0c9c874-rocm10.0.0 with --device /dev/kfd --device /dev/dri, VLLM_CPP_HIP=ON gfx1100.

## Resolution

-
