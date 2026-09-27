ID: ISSUE-GH-1283
Title: **Greedy decode is NOT reproducible at concurrency 16.** Two runs of ONE binary (sha256 `ca114abb…c772ad`) on ONE workload with `--temperature 0 --seed 777 --concurrency 16 --num-prompts 21 --output-len 32`, 23 seconds apart on GB10 / driver `580.173.02` / `VT_ASYNC_RUNNER=0`, emitted DIFFERENT token ids: `5973c5a10a6210085417fb25a29edbd0dc15fe61d7d4f774dd8ff3883dae1d64` (2638 bytes) vs `4cf7923080db6aa29759537f2192f3d9500db11c0e8d72bbb2b4ac6e4614af7c` (2650 bytes). **Not a dedup defect** — `VT_CUDA_GRAPH_DEDUP` is UNSET in both cells; these are the OFF/OFF control of the `ENG-CUDAGRAPH-DEDUP` W5 coarse-key A/B ([#1226](https://github.com/mudler/vllm.cpp/issues/1226)), and the control is the only reason it was seen. Both runs emit `ids_requests=21 ids_total_tokens=672 empty_rows=0`, so the byte delta is JSON decimal width and NOT a length difference, not a truncation and not an early stop; exactly rows 17 and 18 of 21 differ and both diverge MID-DECODE (token index 11 and 5), not at the first token, and both sit in the ragged tail `21 % 16` leaves. Workloads A (conc 24) and C (conc 32) in the same series on the same binary in the same minutes each reproduced themselves exactly, so it is one configuration of three at one repetition each. **The cost: it VOIDS workload B of that run** — `b_off_a == b_exact` and `b_off_a == b_coarse_a` compare against a baseline that does not reproduce itself, and without the OFF/OFF control they would have read as three more byte-identity confirmations. NOT diagnosed beyond the evidence; the issue carries an isolation plan (establish the rate over N repetitions; re-run with `VLLM_CPP_CUDAGRAPH=0` to separate the scheduler from the graph path; a non-ragged `--num-prompts` multiple of `--concurrency`; per-step batch composition for the two diverging requests; the top-2 logit margin at the divergence step, because a near-tie a reduction order can flip is a different defect from a wrong value). Evidence `/mnt/nas_share/rc/dedup-key/logs-ab/ab.log` and `out-ab/ids_b_off_*.json`; owed under `## Owed` of [eng-cudagraph-dedup.md](../../specs/eng-cudagraph-dedup.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1283
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:422`

### Frozen archive evidence

> | [#1283](https://github.com/mudler/vllm.cpp/issues/1283) | — | **Greedy decode is NOT reproducible at concurrency 16.** Two runs of ONE binary (sha256 `ca114abb…c772ad`) on ONE workload with `--temperature 0 --seed 777 --concurrency 16 --num-prompts 21 --output-len 32`, 23 seconds apart on GB10 / driver `580.173.02` / `VT_ASYNC_RUNNER=0`, emitted DIFFERENT token ids: `5973c5a10a6210085417fb25a29edbd0dc15fe61d7d4f774dd8ff3883dae1d64` (2638 bytes) vs `4cf7923080db6aa29759537f2192f3d9500db11c0e8d72bbb2b4ac6e4614af7c` (2650 bytes). **Not a dedup defect** — `VT_CUDA_GRAPH_DEDUP` is UNSET in both cells; these are the OFF/OFF control of the `ENG-CUDAGRAPH-DEDUP` W5 coarse-key A/B ([#1226](https://github.com/mudler/vllm.cpp/issues/1226)), and the control is the only reason it was seen. Both runs emit `ids_requests=21 ids_total_tokens=672 empty_rows=0`, so the byte delta is JSON decimal width and NOT a length difference, not a truncation and not an early stop; exactly rows 17 and 18 of 21 differ and both diverge MID-DECODE (token index 11 and 5), not at the first token, and both sit in the ragged tail `21 % 16` leaves. Workloads A (conc 24) and C (conc 32) in the same series on the same binary in the same minutes each reproduced themselves exactly, so it is one configuration of three at one repetition each. **The cost: it VOIDS workload B of that run** — `b_off_a == b_exact` and `b_off_a == b_coarse_a` compare against a baseline that does not reproduce itself, and without the OFF/OFF control they would have read as three more byte-identity confirmations. NOT diagnosed beyond the evidence; the issue carries an isolation plan (establish the rate over N repetitions; re-run with `VLLM_CPP_CUDAGRAPH=0` to separate the scheduler from the graph path; a non-ragged `--num-prompts` multiple of `--concurrency`; per-step batch composition for the two diverging requests; the top-2 logit margin at the divergence step, because a near-tie a reduction order can flip is a different defect from a wrong value). Evidence `/mnt/nas_share/rc/dedup-key/logs-ab/ab.log` and `out-ab/ids_b_off_*.json`; owed under `## Owed` of [eng-cudagraph-dedup.md](../../specs/eng-cudagraph-dedup.md) | bug |

## Resolution

-
