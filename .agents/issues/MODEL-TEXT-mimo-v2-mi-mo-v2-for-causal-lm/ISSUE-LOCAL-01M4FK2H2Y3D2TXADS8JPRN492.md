ID: ISSUE-LOCAL-01M4FK2H2Y3D2TXADS8JPRN492
Title: ASan heap-use-after-free in MiMoV2 KV-cache write: scaled-V DBuf dies before WriteKvCache reads it; attn_sink Tensor dies before PagedAttention
Row: MODEL-TEXT-mimo-v2-mi-mo-v2-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

The sanitize-cpu ASAN+UBSAN lane is red on test_mimov2_forward (main 41705a7a2). Repro: /tmp/build-sanitize-main with ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 UBSAN_OPTIONS=print_stacktrace=1 VT_POOL_BYPASS=1; log /tmp/sanitize-runs/test_mimov2_forward.log (rc=1). ASan: heap-use-after-free, READ of size 128 in memcpy <- ReshapeAndCacheKernel (src/vt/cpu/cpu_cache.cpp:74) <- vt::ReshapeAndCache (src/vt/ops.cpp:4321) <- dense_attn::WriteKvCache (include/vllm/model_executor/models/kv_cache_route.h:66) <- AttentionBlock (src/vllm/model_executor/models/mimo_v2.cpp:247). The freed 512-byte block was allocated at mimo_v2.cpp:239 (DBuf vs) and freed at mimo_v2.cpp:242 (~DBuf via DevicePool::Put; VT_POOL_BYPASS=1 frees it exactly). Root cause 1: in AttentionBlock, DBuf vs is declared INSIDE the attention_value_scale `if` (mimo_v2.cpp:239) but v3 is reassigned to vs.t() (line 241) and read by dense_attn::WriteKvCache at line 247 AFTER the `if` closes and vs is destroyed. Root cause 2 (same function, same escape pattern, fires once cause 1 is fixed): `Tensor sink = ResidentWeight(...)` is declared inside the `if (w.has_sink_bias)` block (mimo_v2.cpp:267) but `pa.attn_sink = &sink` stores the ADDRESS of that stack Tensor, which dies at the closing brace while vt::PagedAttention (line 275) dereferences it via args.attn_sink->Ptr<float>() (src/vt/cpu/cpu_paged_attn.cpp:138) -- a stack-use-after-scope the test exercises (add_swa_attention_sink_bias=true on SWA layers). The house pattern (deepseek_v4_dsa.cpp:265-271) declares the sink Tensor in the enclosing scope. Fix: hoist `DBuf vs` (default-constructed, move-assigned inside the `if`) and `Tensor sink` to the AttentionBlock scope so both outlive every op that reads them. Numerics-neutral: same ops, same order, same shapes.

## Resolution

Fixed in f1320d752 on row/ci-sanitize-residue-a: DBuf vs (default-constructed, move-assigned in the scale branch) and Tensor sink are hoisted to the AttentionBlock scope so they outlive WriteKvCache and vt::PagedAttention. Verified 2026-10-09: ASAN (CI env, VT_POOL_BYPASS=1) test_mimov2_forward 178/178 with zero AddressSanitizer reports; ASAN battery green (test_kolibri1 234/234, w2 1608/1608, moe_glue 21/21, dequant 10/10, dequant_cache 52/52, mimov2_w1 51/51); Release green (mimov2_forward 178/178, mimov2_w1 51/51, W3 900/900 fingerprints 2.18646/26763 unchanged). Logs /tmp/verify-a-test_mimov2_forward.log, /tmp/verify-a-*.log, /tmp/verify-rel-*.log.
