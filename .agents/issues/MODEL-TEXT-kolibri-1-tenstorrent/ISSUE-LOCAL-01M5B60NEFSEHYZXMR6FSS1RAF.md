ID: ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF
Title: tt-metal upstream head: SFPI 7.88 rvtt_live ICE on the Blackhole layernorm specialization; rms_norm in-model bias persists (worst row regressed 3x)
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-20
Updated: 2026-10-20
Closed: -

## Problem

Probed tt-metal origin/main 2500c1d8f63 (2026-10-19, 1793 commits past our pin) built beside the pinned evidence substrate. Two findings. (1) BLOCKER: the official sfpi_7.88.0_aarch64 tarball (sha256 5fa18268 verified, and a locally-built 7.88) ICEs — 'during GIMPLE pass: rvtt_live ... verify_ssa failed' at __builtin_rvtt_sfppushc(0) in trisck.cc under -flto=1 -flto-partition=one — compiling the layernorm kernel specialization vllm.cpp's rms path configures on Blackhole. SFPI 7.83.0 (local DIY build) compiles it. (2) The rms_norm in-model input-dependent bias is NOT improved upstream and regressed on the largest row: replay of the captured /tmp/normdump rows on upstream head gives tt/host 0.9809 (was 1.0072 on our pin) for normrow-000 (1x2560, max|x|=3.55; max_abs 0.4375, all 2560 elems differ), 1.0033 (was 1.0077) for normrow-004, 0.9970 (was 0.9984) for normrow-001. Upstream switched rsqrt_tile<LEGACY_RSQRT> to ckl::Rsqrt<Approx::Exact> in a new eltwise_chain — did not fix it. Patch audit: aarch64 build + non-temporal memcpy merged upstream; -Werror hw build, Gentoo sfpi, libcody char8_t, and the layernorm override_runtime_arguments (program-cache-hit addr refresh) are STILL NEEDED locally; umd bump superseded. RE-CHECK TRIGGER: SFPI >= 7.89 fixing rvtt_live, or an upstream rms reduce/round composition rework — then rerun the Stage-2 micro replay only (cheap) before any gate. Probe residue: /tmp/tt-metal-new, /tmp/umdtrial-new-install (SFPI 7.83), /tmp/vllm-newsubstrate (4 ad-hoc API-drift shims). Pinned evidence substrate untouched.

## Resolution

-
