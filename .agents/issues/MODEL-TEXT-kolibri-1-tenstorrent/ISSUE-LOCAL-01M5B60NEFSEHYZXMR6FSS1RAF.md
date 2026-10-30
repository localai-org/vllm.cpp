ID: ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF
Title: tt-metal upstream head: SFPI 7.88 rvtt_live ICE on the Blackhole layernorm specialization; rms_norm in-model bias persists (worst row regressed 3x)
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-20
Updated: 2026-10-21
Closed: -

## Problem

Probed tt-metal origin/main 2500c1d8f63 (2026-10-19, 1793 commits past our pin) built beside the pinned evidence substrate. Two findings. (1) BLOCKER: the official sfpi_7.88.0_aarch64 tarball (sha256 5fa18268 verified, and a locally-built 7.88) ICEs — 'during GIMPLE pass: rvtt_live ... verify_ssa failed' at __builtin_rvtt_sfppushc(0) in trisck.cc under -flto=1 -flto-partition=one — compiling the layernorm kernel specialization vllm.cpp's rms path configures on Blackhole. SFPI 7.83.0 (local DIY build) compiles it. (2) The rms_norm in-model input-dependent bias is NOT improved upstream and regressed on the largest row: replay of the captured /tmp/normdump rows on upstream head gives tt/host 0.9809 (was 1.0072 on our pin) for normrow-000 (1x2560, max|x|=3.55; max_abs 0.4375, all 2560 elems differ), 1.0033 (was 1.0077) for normrow-004, 0.9970 (was 0.9984) for normrow-001. Upstream switched rsqrt_tile<LEGACY_RSQRT> to ckl::Rsqrt<Approx::Exact> in a new eltwise_chain — did not fix it. Patch audit: aarch64 build + non-temporal memcpy merged upstream; -Werror hw build, Gentoo sfpi, libcody char8_t, and the layernorm override_runtime_arguments (program-cache-hit addr refresh) are STILL NEEDED locally; umd bump superseded. RE-CHECK TRIGGER: SFPI >= 7.89 fixing rvtt_live, or an upstream rms reduce/round composition rework — then rerun the Stage-2 micro replay only (cheap) before any gate. Probe residue: /tmp/tt-metal-new, /tmp/umdtrial-new-install (SFPI 7.83), /tmp/vllm-newsubstrate (4 ad-hoc API-drift shims). Pinned evidence substrate untouched.

## Resolution

-

## Update 2026-10-21: finding (2) is a compute-config difference, not an upstream regression

The 2026-10-21 re-measurement (P150, docs/bench-evidence/tt-rms-norm-compute-config-20261021.md) corrects the attribution of finding (2) and leaves finding (1) unchanged.

The 0.9809 reading is not an upstream regression and the rsqrt rework is not its cause. A standalone ttnn probe replays the same captured rows with an explicitly chosen compute-kernel configuration and reads 0.980947 with fp32_acc=false and 1.007197 with fp32_acc=true and approx_mode=false, on the upstream substrate and on the pinned substrate alike. Upstream main keeps `approx_mode = true` and `fp32_acc = false` in `rmsnorm_default_compute_config`; the pinned fork carries a local edit that flips both, and the vllm.cpp rms path passes no compute configuration, so the substrate default decides. Upstream main reproduces the pinned 1.0072 exactly when the same configuration is requested, so the 1793 upstream commits changed no measured value in the rms path on these rows.

Consequence for an upstream filing: the claim to make is that the fused rms_norm kernel's deviation from a host-f64 oracle depends on its compute configuration and on the input row, and that neither the default configuration nor the f32 reduction removes it. The claim not to make is that upstream's rms path regressed or that the rsqrt rework changed the result.

Finding (1), the SFPI 7.88 rvtt_live ICE, is untouched by this measurement and still blocks a build on the official 7.88.0 tarball. The upstream numbers here carry the SFPI 7.83.0 local-build caveat.

## Update 2026-10-21: SFPI 7.89.0 re-check, no ICE; the ICE belongs to the locally built 7.88.0

The re-check trigger fired. The official sfpi_7.89.0_aarch64_debian.txz (sha256 78b4460a697966393f8a0ccf18b071ee610b768650ffbecadc4a9e066b0fb9a7, digest equal to the GitHub release asset) does NOT hit the rvtt_live/verify_ssa ICE. The tt-metal tree at origin/main 2500c1d8f63 builds clean against 7.89.0 (1437/1437 targets, exit 0, 10 min 42.7 s wall at -j 8), the Blackhole layernorm compute-kernel specialization compiles and LTO-links with 7.89.0, and the device rms micro runs with the same numbers it produces under 7.88.0 (tt/host2 0.980947, 1.0033, 0.996982 on normrow-000, -004, -001).

The measurement corrects the attribution above. The ICE reproduces with the LOCALLY BUILT 7.88.0 toolchain (riscv-tt-elf-g++ tenstorrent/sfpi-DIY:7.88.0) and not with the OFFICIAL 7.88.0 release tarball (tenstorrent/sfpi:7.88.0[1039]). Same tree, same kernel, same flags, same harness, one toolchain swapped: 3 ICE sites in the locally built arm, none in the official arm. The ICE text is unchanged: "during GIMPLE pass: rvtt_live ... verify_ssa failed" at __builtin_rvtt_sfppushc (0) in trisck.cc:65, one site per TRISC. The earlier statement that the official tarball ICEs is not supported by this measurement, so an upstream report must name the locally built toolchain or another 7.88.0 build.

Evidence: docs/bench-evidence/tt-metal-sfpi-789-layernorm-compile-20261021.md. The toolchain blocker on the pin advance is GONE; the rms_norm in-model bias is unchanged by the toolchain (kernel numerics, compiler independent) and this issue stays OPEN for that part.

