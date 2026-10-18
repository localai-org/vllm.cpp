ID: ISSUE-LOCAL-01M4FK2A9T7BPB0KHPAA8TN1VA
Title: ASan heap-use-after-free in kolibri1 RoPE path: qk-norm DBufs die before RoPE/KV-write read their views
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

The sanitize-cpu ASAN+UBSAN lane is red on test_kolibri1_w2 (main 41705a7a2). Repro: /tmp/build-sanitize-main with ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 UBSAN_OPTIONS=print_stacktrace=1 VT_POOL_BYPASS=1; log /tmp/sanitize-runs/test_kolibri1_w2.log (rc=1). ASan: heap-use-after-free, READ of size 2 in vt::LoadUnaligned<unsigned short> (include/vt/unaligned.h:29) <- LoadF32 (src/vt/cpu/cpu_ops.cpp:43) <- RopeRotateHead (src/vt/cpu/cpu_ops.cpp:1313) <- RopeNeoxKernel lambda <- vt::RopeNeox (src/vt/ops.cpp:1814) <- AttentionBlock (src/vllm/model_executor/models/kolibri1_forward.cpp:260). The freed 1280-byte block was allocated at kolibri1_forward.cpp:243 (DBuf qn) and freed at kolibri1_forward.cpp:250 (~DBuf via DevicePool::Put, VT_POOL_BYPASS=1 frees it exactly). Root cause: in AttentionBlock the per-head qk-norm DBufs qn/kn are declared INSIDE the inner {} block (kolibri1_forward.cpp:243-244), but q3/k3 are reassigned to Reshape views of their storage (lines 248-249) and used AFTER the block closes: vt::RopeNeox(d.q, q3, k3, ...) at line 260 and dense_attn::WriteKvCache(k3/v3) later. The DBuf destructor returns the block to the DevicePool at the closing brace; with VT_POOL_BYPASS=1 the pool frees it, so RoPE (and the KV write) read freed memory. Without the bypass the pool recycles the block, so the same defect is a silent cross-op corruption. The TSAN lane independently reports heap-use-after-free on the same test. The RoPE kernel itself is innocent: the caller lets the owner die before the ops that read its views run (the house pattern — kimi_linear_device.cpp:708-717 — declares qn/kn in the enclosing scope). Fix: hoist the qn/kn DBuf declarations out of the inner block so they live until AttentionBlock returns. Numerics-neutral: same ops, same order, same shapes.

## Resolution

Fixed in 5bb9d85ff on row/ci-sanitize-residue-a: the qn/kn DBufs are hoisted to the AttentionBlock scope so they outlive vt::RopeNeox and the KV-cache write. Verified 2026-10-09: ASAN (ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 UBSAN_OPTIONS=print_stacktrace=1 VT_POOL_BYPASS=1) test_kolibri1_w2 1608/1608 with zero AddressSanitizer reports; TSAN (VT_POOL_BYPASS=1) 1608/1608 with zero ThreadSanitizer reports; Release battery green (test_kolibri1 234/234, w2 1608/1608, decode_bench anchor 109726, W3 900/900 with 141/145 argmax chain and fingerprints 2.18646/26763 unchanged). Logs /tmp/verify-a-test_kolibri1_w2.log, /tmp/verify-tsan-kolibri1_w2.log, /tmp/verify-rel-*.log.
