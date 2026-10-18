ID: ISSUE-LOCAL-01M4D5W0HSEPNP26AT30CEP874
Title: TT build: any binary crashes under an empty environment — the static-init DeviceAvailable() probe requires TT_METAL_RUNTIME_ROOT (witness: kolibri1 dequant-cache default-off helper)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: -

## Problem

Found during the kolibri-tt B2b-i bring-up (2026-10-08): in a VLLM_CPP_TENSTORRENT=ON build, EVERY binary that links libvllm.a segfaults (exit 139) when run with an empty environment (env -i). The chain: the TT ops registrar's static initializer calls vt::tenstorrent::DeviceAvailable() → tt::tt_metal::GetNumAvailableDevices() → MetalContext::create_default_instance → RunTimeOptions root-dir resolution → TT_FATAL 'Root Directory is not set' (rtoptions.cpp:363) → the fatal handler itself crashes without the runtime env. Witness: tests/vllm/models/test_kolibri1_dequant_cache.cpp's default-off probe fork/execs its helper binary with envp={nullptr} (to prove VT_KOLIBRI1_DEQUANT_CACHE_MB is unset); in a TT build the helper dies with SIGSEGV before main, so the case fails (6/7 pass, the probe is the 1 failure). The same crash hits test_kolibri1_tt, test_kolibri1, and any other TT-build binary under env -i. This is pre-existing (the registrar's DeviceAvailable() static-init probe predates the B2b-i slice; no B2b-i change adds a static initializer) — the B2b-i work merely ran the CPU test family inside a TT build and exposed it. The CPU-only build is unaffected (no TT backend linked, no probe). Impact: any TT-build binary run with a scrubbed environment crashes at startup; the kolibri1 dequant-cache default-off case cannot pass in a TT build. The fix belongs to the TT backend (e.g. DeviceAvailable() probing without MetalContext init, or tolerating a missing root dir at static-init time), not to the kolibri rows.

## Resolution

-
