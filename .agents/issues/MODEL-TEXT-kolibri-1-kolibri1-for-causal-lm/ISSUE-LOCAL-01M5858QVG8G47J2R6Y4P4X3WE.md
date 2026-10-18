ID: ISSUE-LOCAL-01M5858QVG8G47J2R6Y4P4X3WE
Title: cpu_paged_attn_bfdot_arm.cpp includes arm_neon.h unconditionally — breaks every non-Arm build
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-18
Closed: 2026-10-18

## Problem

The draft-arm commit added src/vt/cpu/cpu_paged_attn_bfdot_arm.cpp with an unconditional '#include <arm_neon.h>' at line 30, outside the file's own __aarch64__/__ARM_FEATURE_BF16 guard. Every x86 and Windows CMake configure compiles the TU from the unconditional source list and dies: 'arm_neon.h: No such file or directory' (verify cpu linux/amd64, run 39150632927, job 117746862712, pr=3464). The repository's established shape (cpu_ops.cpp:24, cpu_paged_attn.cpp:16, cpu_quant_dot_arm.cpp on main) puts the include inside the arch guard. Also exposed: the stub TU uses uint16_t with no <cstdint> include, which only compiled by accident through arm_neon.h. Repair: move the include inside the guard, add <cstdint> outside it; x86 syntax-only compile green with -Wall -Wextra -Werror.

## Resolution

2026-10-18: include moved inside the arch guard, <cstdint> added for the stub TU; x86 -Wall -Wextra -Werror syntax-only compile green; Arm bf16 tier unchanged (same guard, same CMake property, same runtime gate). Rides PR #3464.
