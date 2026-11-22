ID: ISSUE-LOCAL-01M80WA9T38SWMMWYDJ9T938K8
Title: NEW-CANDIDATE-PORTS W0 (inkling2_loop): tabulate ggml Q4_1/Q5_1 in the GGUF reader and land the inkling2_loop parser + refusing loader row
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: -

## Problem

The pinned mudler/Inkling-2-Mini-APEX-GGUF Balanced artifact (arch inkling2_loop, 704 tensors, 59 KVs, 14 type-7 ffn_down_exps) makes GgufFile::Open throw 'gguf: tensor "blk.10.ffn_down_exps.weight" has unknown ggml type id 7' because FindGgmlTraits (src/vllm/model_executor/model_loader/gguf_reader.cpp:200) jumps case 6 -> case 8 and never tabulates ggml id 7 (Q5_1, block 32 elems / 21 bytes) or id 3 (Q4_1, 32/20). The artifact also has no config parser or loader row: HfConfigFromGgufDispatch refuses it as an unrecognized architecture. W0 delivers the two type-table rows with a real-reader unit test over the pinned artifact path (77-style skip when absent) and an inkling2_loop config parser plus one loader row that parses and then refuses at the forward boundary naming NEW-CANDIDATE-PORTS, following the NEW-CANDIDATE-PARSERS pattern (olala/hy_v4_flash). Upstream oracle vLLM PR #69028 (merged 2026-11-21) adds Inkling2LoopForConditionalGeneration / model_type inkling2_loop with loop_config fields mirrored here. The vLLM pin advance is OWED and out of scope (recorded in the spec). The broader enumeration/support path is tracked by ISSUE-LOCAL-01M7ZYAY4W7STQCBPSF6NQRF78; this issue owns W0's concrete steps (1) and (2).

## Resolution

-
