ID: ISSUE-LOCAL-01M3Z1KXM2EQ11N8BXFNV1Z7RY
Title: release archive validation fails: vllm-server embeds the build path
Row: ROAD-V1-RELEASE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-02
Updated: 2026-10-02
Closed: -

## Problem

Docker release lanes (e.g. CI run 37034253522, verify vulkan linux/amd64) fail the deterministic-archive validator: 'archive file bin/vllm-server embeds forbidden build path /src/build-container-vulkan'. The FetchContent parakeet.cpp dependency (added by 2833d6300, 2026-09-27) compiles vendored ggml sources that live under CMAKE_BINARY_DIR/_deps, so their __FILE__/assert strings embed the absolute build directory; the pre-existing --forbid-path rule (426c2dd52, 2026-08-09) correctly rejects the artifact. Fix at the source: -ffile-prefix-map for the source and binary directories so no absolute build path is embedded.

## Resolution

2026-10-02: fixed by -ffile-prefix-map in CMakeLists.txt (row/release-path-scrub); local before/after scan 16 -> 0 embedded build-path occurrences; validator unit tests green.
