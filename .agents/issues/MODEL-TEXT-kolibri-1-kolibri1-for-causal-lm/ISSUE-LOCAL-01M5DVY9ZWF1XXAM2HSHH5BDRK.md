ID: ISSUE-LOCAL-01M5DVY9ZWF1XXAM2HSHH5BDRK
Title: the default Kolibri-1 dequant-cache budget is sampled at the wrong moment and can land below the measured knee
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

The default budget of the Kolibri-1 CPU fp8 dequant cache (src/vllm/model_executor/models/kolibri1_dequant_cache.h) is planned on the FIRST USE of the process-wide cache, which happens DURING the 74 GB checkpoint load: the page cache is full by then, so /proc/meminfo MemAvailable reads 66 GiB where the host's idle reading is ~145 GiB. The policy clamp(MemAvailable / 8, 1 GiB, 32 GiB) then yields 8.25 GiB, and a budget that low buys nothing: measured on node 3 (aarch64, 3.4 GHz, 32 threads, empty-prompt 32-token decode, 2026-10-21) the default leg showed budget_mib=8454 from budget_source_mib=67638, hits=9097, misses=18349, evictions=16068, a 33% hit rate and 3.84 tok/s, against 6.38 tok/s for an explicit 16384 MiB budget and 3.62 tok/s with the cache off. The defect has two parts with one root cause: the reading is taken at a moment our own load has depressed, and the policy can land below the 16 GiB knee the measurement puts at the point where the cache stops thrashing. Fix: sample the host's available memory at PROCESS START (a namespace-scope initialiser, before main and before any model load) and use the MAXIMUM of that reading and the reading at plan time, and floor the budget at the measured 16 GiB knee when the host's honest reading is at or above a stated host threshold, keeping the 1 GiB floor, the 32 GiB cap, the env contract (0 disables, N sets exactly N MiB, an unparseable value falls back to the default policy) and the byte-for-byte uncached fallback.

## Resolution

2026-10-21: FIXED. The default budget is planned from the HONEST available-memory
reading, the maximum of the PROCESS-START reading (a namespace-scope
initialiser, taken before main and before the 74 GB checkpoint load) and the
reading at plan time, and the policy is:

    honest_bytes = max(start_bytes, plan_bytes)
    honest_bytes == 0              -> budget = 16 GiB   (UNKNOWN)
    honest_bytes < 16 GiB + 8 GiB  -> budget = 0        (DISABLED)
    otherwise                      -> budget = clamp(honest_bytes / 8,
                                                        16 GiB, 32 GiB)

The KNEE (16 GiB) is the FLOOR for every host that can afford it. A host that
reports less than the knee plus an 8 GiB reserve cannot afford the knee at
all, so the cache is DISABLED and the forward runs the uncached path, byte for
byte: a sub-knee cache is WORSE than useless, so the cache either runs at or
above the knee or it does not run. An unreadable report uses the knee, because
an unknown budget is not a reason to refuse and not a reason to hand out a
sub-knee budget. The env contract is unchanged (0 disables, N sets exactly N
MiB, an unparseable value falls back to the default policy) and the uncached
fallback is byte-for-byte.

RED-FIRST: the new gate case (a load-depressed reading must not land below the
measured knee) states the knee as a literal, so it compiles against the
pre-repair policy and the red is an ASSERTION FAILURE rather than a compile
error. A probe compiled against the base commit's header (84d98a9e6) reports
budget_bytes=8858370048 (8.25 GiB) for the load-depressed 66 GiB reading and
exits 1; the repaired policy reports budget_bytes=17179869184 (16.00 GiB) and
exits 0. The policy case also pins the disable rule (a host below the knee
plus the reserve gets budget 0) and the UNKNOWN rule (the knee, never a
refusal and never a sub-knee budget).

MEASURED after the repair (2026-10-21, node 3, aarch64, 3.4 GHz, 32 threads,
numactl --cpunodebind=3 --interleave=all, VLLM_CPP_CPU_THREADS=32,
/tmp/build-dq/tests/test_kolibri1_decode_bench with VT_KOLIBRI1_PROFILE=1, the
file mutex $HOME/gpu.lock held for the window):

  leg              budget_MiB  hits    misses  evictions  hit rate  tok/s
  default r1       23143       71717   18229   10072      79.7%     7.58
  default r2       27107       72184   17762    8019      80.3%     7.36
  default r3       26971       72169   17777    8089      80.2%     7.50
  16384 MiB r1-3   16384       71097   18849   13396      79.0%     7.15 / 7.23 / 7.14
  0 (off) r1-3     0                0       0        0       0.0%     3.49 / 3.58 / 2.45

The default leg is enabled, its budget is 22.6-26.5 GiB (at or above the 16
GiB knee), its hit rate is ~80% (the defect's 33%), and its throughput
(7.36-7.58 tok/s) is above the 6.38 tok/s the finding measured for an explicit
16 GiB budget and above the same-binary explicit 16 GiB legs (7.14-7.23
tok/s). The gap between the plan-time reading (140-144 GiB) and the
process-start reading (181-212 GiB) IS the checkpoint load depressing the
host's own report. The off leg confirms the uncached fallback: enabled=0, no
cache traffic, and the SAME token chain (11622...) as the cached legs. The
last off leg ran under a load average of 53.9 (another build on the box) and
is the low outlier.

GATES: test_kolibri1_dequant_cache 13/13 (66150 assertions), test_kolibri1
27/27, test_kolibri1_w2 7/7, test_kolibri1_numa 3/3, test_kolibri1_dequant
3/3, test_kolibri1_moe_glue 9/9, test_kolibri1_bfdot_gemv 5/5,
test_kolibri1_decode_bench SUCCESS. The W3 token gate is red on pristine
origin/main and reproduces there with the SAME numbers as on this tree (134/138
compared positions, 4 near-tie flips, 0 hard flips, worst topk diff 3.40201,
worst sum diff 87462). NO NUMERICS CHANGE: the cached bytes stay
byte-identical to the production uncached dequant, which the gate pins cold
and hot, and the cached and uncached legs produce the same token chain.

Docs: .agents/specs/kolibri-1-cpu.md, docs/ENVIRONMENT.md and
docs/FEATURES.md carry the corrected policy, the measured knee, and the honest
statement that a sub-knee budget is worthless.
