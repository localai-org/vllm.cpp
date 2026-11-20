ID: ISSUE-LOCAL-01M7W2ZKR8HRBYJF6HCPP46AGH
Title: TT public surface: EnsureHost forward-declared in always-compiled drafter test breaks CPU-only link; guard the stub discipline with a checker
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-20
Updated: 2026-11-20
Closed: 2026-11-20

## Problem

tests/vllm/models/test_qwen38_dflash2_drafter_tt.cpp:74-77 forward-declares vt::tenstorrent::EnsureHost because tenstorrent_internal.h pulls ttnn headers, and calls it at :787. EnsureHost is declared only in tenstorrent_internal.h and defined only in tenstorrent_residency.cpp (TT-only TU), so every CPU-only lane (build-test-cpu, build-newest-gcc, cpu-arm64-full) that links this unconditionally-registered test fails with an undefined reference to vt::tenstorrent::EnsureHost. Same hazard class: tenstorrent_device.h declares RegisterHostBuffer/UnregisterHostBuffer/MarkHostWritten/MarkScratchAcquired/EnsureHostBytes/CopyDeviceDeviceIfCapture/MemsetDeviceIfCapture/MemsetDeviceFill/CopyDeviceDeviceIfResident UNCONDITIONALLY but they are defined only in TT-only TUs — any always-compiled TU that references them fails the same way. The established pattern (tenstorrent_device.h:142-166, the WarmPagedKvShadow inline no-op stubs) keeps device-agnostic TUs linkable precisely by giving every public symbol an #else stub; the litter is symbols that skipped it. Fix: (1) move EnsureHost into the public header with #else no-op stubs and include that header from the test instead of forward-declaring; (2) give the unconditional danglers #else stubs too; (3) add scripts/check-tt-public-surface.py (picked up by the preflight sweep) enforcing: always-compiled TUs (src minus TT-only, include/, examples/, tests registered outside if(VLLM_CPP_TENSTORRENT)) may reference vt::tenstorrent::X only if X has a non-TT-branch declaration in tenstorrent_device.h, may not include internal TT headers, may not re-open namespace vt::tenstorrent to forward-declare; and every symbol declared in the public header must have a non-TT-branch definition. Red-before: the checker run on the pre-fix tree names the drafter test; green-after plus a CPU-only link of test_qwen38_dflash2_drafter_tt in /tmp/build-parsers-review.

## Resolution

Fixed on branch fix/ensurehost-cpu-link (a5b7b18fa spec + 7880f6528 fix, on main 14e06b7bc). EnsureHost moved to the public tenstorrent_device.h with #else no-op stubs; 17 unconditional header danglers (DeviceAvailable; SharedMeshDevice moved to a TT-only section; RegisterHostBuffer/UnregisterHostBuffer/MarkHostWritten/MarkScratchAcquired/EnsureHostBytes; CopyDeviceDeviceIfCapture/MemsetDeviceIfCapture/MemsetDeviceFill/CopyDeviceDeviceIfResident; 6 Trace* entry points) now carry stubs; the drafter test includes the public header instead of forward-declaring. New gate scripts/check-tt-public-surface.py (picked up by the preflight sweep) enforces R1-R4 over always-compiled TUs; red-before on the pre-fix tree was 19 violations (drafter:74 R2, drafter:787 R3 EnsureHost, 17 R4), green-after 0 violations / 96 stubbed symbols; unit tests 9/9. Link proof: CPU-only build (VLLM_CPP_TENSTORRENT=OFF) of test_qwen38_dflash2_drafter_tt in /tmp/build-tt-surface BUILD EXIT 0, nm shows W (weak inline) EnsureHost, binary runs 5/5 cases 143/143 assertions on CPU. Spec .agents/specs/tt-public-surface-guard.md committed first (a5b7b18fa). Trailer/style gates OK on both commits.
