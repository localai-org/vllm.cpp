ID: ISSUE-GH-1636
Title: **`KV-FP8` W1's three read-side comments anchor `scaled_vec_conversion<float, uint8_t>` at `quant_utils.cuh:302-308`, which at pin `555967922` is the IDENTITY primary template plus the header of the fp8->HALF specialization.** Lines 301-305 are `template <typename Tout, typename Tin> ... { return x; }` and 307-314 are the `<uint16_t, uint8_t>` conversion; the `<float, uint8_t>` one the comments describe is at `:419-429` under the `// fp8 -> float` label at `:418`. Sites, all landed by W1 and all outside the W2 change's authority: `include/vt/fp8_kv.h:92`, `include/vt/ops.h:1129`, `src/vt/cpu/cpu_paged_attn.cpp:164`. W2 ([#1593](https://github.com/mudler/vllm.cpp/issues/1593), PR [#1606](https://github.com/mudler/vllm.cpp/pull/1606)) copied the same wrong anchor into four new places and CORRECTED all four there; these three are filed rather than fixed in flow. Same shape, second anchor: `Fp8KVCacheDataType` is cited at `dtype_fp8.cuh:9-13`, which is the `#include <cuda_fp8.h>` guard -- the enum is at `:15-19` (`include/vt/fp8_kv.h:5`, `:30`). Third, a different kind: `.agents/engine-matrix.md` and `.agents/quantization-matrix.md` both say the W2 CUDA translation units are UNCOMPILED, and CI job `cuda-fat-build` built them for ten architectures under `-Werror=all-warnings` and PASSED on `4d71e776efc18cb5e61a26e642ddad8de5339134` (run 32495320287, job 96812232428). What stays true is that nothing has been EXECUTED on a device, because that job configures `-DVLLM_CPP_BUILD_TESTS=OFF`; both clauses need the narrower statement. An upstream anchor is how the next reader checks a port against the oracle, and one that lands on a `return x;` primary template invites the conclusion that the port is unfaithful. Listed under `## Owed` in [fp8-kv-cache.md](../../specs/fp8-kv-cache.md)
Row: KV-FP8
State: UNKNOWN
Kind: bug
GitHub: 1636
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:571`

### Frozen archive evidence

> | [#1636](https://github.com/mudler/vllm.cpp/issues/1636) | `KV-FP8` | **`KV-FP8` W1's three read-side comments anchor `scaled_vec_conversion<float, uint8_t>` at `quant_utils.cuh:302-308`, which at pin `555967922` is the IDENTITY primary template plus the header of the fp8->HALF specialization.** Lines 301-305 are `template <typename Tout, typename Tin> ... { return x; }` and 307-314 are the `<uint16_t, uint8_t>` conversion; the `<float, uint8_t>` one the comments describe is at `:419-429` under the `// fp8 -> float` label at `:418`. Sites, all landed by W1 and all outside the W2 change's authority: `include/vt/fp8_kv.h:92`, `include/vt/ops.h:1129`, `src/vt/cpu/cpu_paged_attn.cpp:164`. W2 ([#1593](https://github.com/mudler/vllm.cpp/issues/1593), PR [#1606](https://github.com/mudler/vllm.cpp/pull/1606)) copied the same wrong anchor into four new places and CORRECTED all four there; these three are filed rather than fixed in flow. Same shape, second anchor: `Fp8KVCacheDataType` is cited at `dtype_fp8.cuh:9-13`, which is the `#include <cuda_fp8.h>` guard -- the enum is at `:15-19` (`include/vt/fp8_kv.h:5`, `:30`). Third, a different kind: `.agents/engine-matrix.md` and `.agents/quantization-matrix.md` both say the W2 CUDA translation units are UNCOMPILED, and CI job `cuda-fat-build` built them for ten architectures under `-Werror=all-warnings` and PASSED on `4d71e776efc18cb5e61a26e642ddad8de5339134` (run 32495320287, job 96812232428). What stays true is that nothing has been EXECUTED on a device, because that job configures `-DVLLM_CPP_BUILD_TESTS=OFF`; both clauses need the narrower statement. An upstream anchor is how the next reader checks a port against the oracle, and one that lands on a `return x;` primary template invites the conclusion that the port is unfaithful. Listed under `## Owed` in [fp8-kv-cache.md](../../specs/fp8-kv-cache.md) | bug |

## Resolution

-
