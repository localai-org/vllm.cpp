ID: ISSUE-LOCAL-01M3QDT6A8JEM8WNJTPR16MXHR
Title: Every Marlin NVFP4 repack leaks its packed and scale device buffers, because Nvfp4Weight publishes one allocation on two owning handles
Row: LOAD-MODELOPT-NVFP4-BORROW
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

Nvfp4Weight carries TWO owning handles per buffer: the type-specific d_packed/d_scale members (include/vllm/model_executor/models/qwen3_5_weights.h:707-708) and the generic raw-twin slots packed.d_dev/scale.d_dev (qwen3_5_weights.h:195) that AdoptDeviceBytesAsHost keys on (src/vllm/model_executor/models/qwen3_5_weights.cpp:420-429). ResidentNvfp4 publishes the SAME allocation on both: w.d_packed = shared_ptr(p, Free) followed by w.packed.d_dev = w.d_packed (dense_nvfp4_gemm.h:319-347, twin at src/vllm/model_executor/models/qwen3_5.cpp:1449). Releasing a repacked weight means dropping both handles, but every Marlin repack builder drops only the type-specific pair: dense_nvfp4_gemm.h:448-449, dense_nvfp4_gemm.h:667-670 (both operands of the pair), qwen3_5.cpp:2952-2953, qwen3_5.cpp:3132-3135, qwen3_5.cpp:6890-6891 and laguna.cpp:650-655. The surviving packed.d_dev alias holds the control block, so each repacked weight keeps a full packed+scale device copy alive for the process lifetime; on an MoE model that is one leak per expert per projection. The repack is precisely the step that is supposed to keep peak weight memory flat (dense_nvfp4_gemm.h:395-397: "we do this repack lazily on first forward and then FREE the fp4 originals"), so the leak silently undoes the memory design on exactly the cards that need it: the NVFP4 arms are served on 24 GiB consumer Blackwell (sm_120a), where a second full copy of the fp4 originals does not fit beside the repacked weights. Cause is grounded in the source above, not inferred from a failed allocation.

## Resolution

-
