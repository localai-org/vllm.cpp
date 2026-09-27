ID: ISSUE-GH-1250
Title: No speed number exists for `NemotronHForCausalLM` on any axis, and one is now takeable: the A3 token gate reads `TOKEN MATCH: 96/96 over 3 prompt(s) (full rows=3, short rows=0, mode=decode)` `STRICT PASS` on GB10 `sm_121a` under the [#1157](https://github.com/mudler/vllm.cpp/issues/1157) fix, against 4/24 on the same binary with the fix reverted, on the released `nemotron-3.5-lightning-30b-nvfp4` at revision `29f2d1746d8f41e316523194b19018707749b1b1` | the gate's own legs measured 264.4 s engine load and 342.61 / 328.19 / 327.51 s for 32 tokens each, about 10.3 s per output token at batch 1, and the two arms that cause it are NAMED and unlanded: the NVFP4 `lm_head` still executes HOST-side because `src/vllm/model_executor/models/nemotron_h.cpp::NemotronHHostLmHead` refuses it on a non-CPU queue (A2-Q2b), and the 46 FP8 W8A8 mamba `in_proj`/`out_proj` projections, 36.6% of decode bytes and 27.6% of GEMM FLOPs, still execute host-side (A2-Q1, PR [#1289](https://github.com/mudler/vllm.cpp/pull/1289); [#940](https://github.com/mudler/vllm.cpp/issues/940) CLOSED) | two comparability facts the measurement controls for: `--gpu-memory-utilization 0.92` does NOT size our KV pool because the profile run is unimplemented ([#83](https://github.com/mudler/vllm.cpp/issues/83)) and it falls back to 256 blocks, so the pool is stated explicitly on our side and matched on the oracle's; and the oracle has never RUN a model inside a lease ([#1185](https://github.com/mudler/vllm.cpp/issues/1185) measured build, install, import and `torch.cuda.is_available()` only, and the recorded failure mode of the step after `torch.compile` on this host is a REBOOT of the box), so an oracle that cannot run yields numerator-only figures recorded as UNGATED and never as a ratio. Spec [`nemotron-h-speed-parity.md`](../../specs/nemotron-h-speed-parity.md)
Row: MODEL-NEMOTRON-H-ABI-A2P
State: UNKNOWN
Kind: perf
GitHub: 1250
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:477`

### Frozen archive evidence

> | [#1250](https://github.com/mudler/vllm.cpp/issues/1250) | `MODEL-NEMOTRON-H-ABI-A2P` | No speed number exists for `NemotronHForCausalLM` on any axis, and one is now takeable: the A3 token gate reads `TOKEN MATCH: 96/96 over 3 prompt(s) (full rows=3, short rows=0, mode=decode)` `STRICT PASS` on GB10 `sm_121a` under the [#1157](https://github.com/mudler/vllm.cpp/issues/1157) fix, against 4/24 on the same binary with the fix reverted, on the released `nemotron-3.5-lightning-30b-nvfp4` at revision `29f2d1746d8f41e316523194b19018707749b1b1` \| the gate's own legs measured 264.4 s engine load and 342.61 / 328.19 / 327.51 s for 32 tokens each, about 10.3 s per output token at batch 1, and the two arms that cause it are NAMED and unlanded: the NVFP4 `lm_head` still executes HOST-side because `src/vllm/model_executor/models/nemotron_h.cpp::NemotronHHostLmHead` refuses it on a non-CPU queue (A2-Q2b), and the 46 FP8 W8A8 mamba `in_proj`/`out_proj` projections, 36.6% of decode bytes and 27.6% of GEMM FLOPs, still execute host-side (A2-Q1, PR [#1289](https://github.com/mudler/vllm.cpp/pull/1289); [#940](https://github.com/mudler/vllm.cpp/issues/940) CLOSED) \| two comparability facts the measurement controls for: `--gpu-memory-utilization 0.92` does NOT size our KV pool because the profile run is unimplemented ([#83](https://github.com/mudler/vllm.cpp/issues/83)) and it falls back to 256 blocks, so the pool is stated explicitly on our side and matched on the oracle's; and the oracle has never RUN a model inside a lease ([#1185](https://github.com/mudler/vllm.cpp/issues/1185) measured build, install, import and `torch.cuda.is_available()` only, and the recorded failure mode of the step after `torch.compile` on this host is a REBOOT of the box), so an oracle that cannot run yields numerator-only figures recorded as UNGATED and never as a ratio. Spec [`nemotron-h-speed-parity.md`](../../specs/nemotron-h-speed-parity.md) | perf |

## Resolution

-
