ID: ISSUE-LOCAL-01M7W6RZN939E9XT39ZK7EME1Q
Title: macos-metal-mlx red on main: cpu_quant_dot_q6k_neon.cpp:79 uses vshrq_n_u8 with a variable shift - clang requires a constant
Row: GATE-METAL-MLX-COMPILE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-20
Updated: 2026-11-20
Closed: 2026-11-20

## Problem

macos-metal-mlx is red on main (run 44692868221) compiling src/vt/cpu/cpu_quant_dot_q6k_neon.cpp:79: "error: argument to '__builtin_neon_vshrq_n_v' must be a constant integer", at the line vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, shift), vdupq_n_u8(3)), 4) inside the unpack lambda — shift is a runtime parameter (called with 0/2/4/6). gcc-aarch64 accepts it (build-test-cpu-arm64-full reached the link stage), clang 17 on macOS does not. Fix: use the variable-shift form vshrq_u8(qhv, vdupq_n_u8(shift)), matching how the sibling q4k NEON kernel handles per-lane shifts. Verify with clang --target=aarch64 -fsyntax-only when clang is available, else mirror the q4k idiom exactly and rely on the arm64 CI lane.

## Resolution

Fixed on branch fix/q6k-neon-variable-shift (pushed to mine): the unpack lambda now uses vshlq_u8(qhv, vdupq_n_s8(-shift)) — the ACLE variable-shift form (negative = right); vshrq_n_u8 requires a constant and clang rejected the runtime shift at cpu_quant_dot_q6k_neon.cpp:79. Red-before reproduced locally with clang++ --target=aarch64-linux-gnu -march=armv8.2-a+dotprod+fp16 (same :79 error); green-after the file and cpu_quant_dot_q4k_neon.cpp syntax-check clean as aarch64. macos-metal-mlx CI on the PR is the final gate.
