ID: ISSUE-LOCAL-01M7W6RZN939E9XT39ZK7EME1Q
Title: macos-metal-mlx red on main: cpu_quant_dot_q6k_neon.cpp:79 uses vshrq_n_u8 with a variable shift - clang requires a constant
Row: GATE-METAL-MLX-COMPILE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-20
Updated: 2026-11-20
Closed: -

## Problem

macos-metal-mlx is red on main (run 44692868221) compiling src/vt/cpu/cpu_quant_dot_q6k_neon.cpp:79: "error: argument to '__builtin_neon_vshrq_n_v' must be a constant integer", at the line vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, shift), vdupq_n_u8(3)), 4) inside the unpack lambda — shift is a runtime parameter (called with 0/2/4/6). gcc-aarch64 accepts it (build-test-cpu-arm64-full reached the link stage), clang 17 on macOS does not. Fix: use the variable-shift form vshrq_u8(qhv, vdupq_n_u8(shift)), matching how the sibling q4k NEON kernel handles per-lane shifts. Verify with clang --target=aarch64 -fsyntax-only when clang is available, else mirror the q4k idiom exactly and rely on the arm64 CI lane.

## Resolution

-
