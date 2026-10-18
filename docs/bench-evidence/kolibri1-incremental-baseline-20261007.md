# Kolibri-1 CPU incremental-decode production shape — committed baseline and
# the block-table question (MODEL-TEXT-kolibri-1)

Date: 2026-10-07. Host: aarch64, 128 cores, 255 GB RAM, Linux (same box as
[kolibri1-dequant-cache-negative-20261007.md](kolibri1-dequant-cache-negative-20261007.md)).
Tree: pristine `origin/main` 323f81ca6, worktree `/tmp/vllm-kolibri-bench`,
branch `row/kolibri-bench`, build Release with
`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TENSTORRENT=OFF`, all runs
`VLLM_CPP_CPU_THREADS` as noted.

## The open question: the morning block-table crash

The 2026-10-07 morning scratch bench crashed at
`src/vt/cpu/cpu_paged_attn.cpp` ("the block table is shorter than the
sequence it must address"). VERDICT: **harness misconfiguration, no product
bug.** The crashing harness carried the W3 topology's 8-block table
(`kBlockSize = 16`, `kNumBlocks = 8` → 128 slots) while the production shape
must address 128 prompt tokens + 64 greedy tokens = 192 positions, which
needs 12 logical blocks. The refusal at the check is the correct refusal: the
caller asked the table to address a sequence it has no columns for. A
properly-configured engine (identity block table, 16 columns × 16-slot
blocks, 12 of 16 blocks used) drives the full production shape through the
same public entry points the W3 harness uses (`ForwardKolibri1Forward` over
`CommonAttentionMetadata` + `MultiKvCacheIndex`) to completion on pristine
main. Nothing landed in `src/` in this unit; the fix is the harness's table
sizing, which is what the committed bench does.

## Production-shape baseline (committed bench:
`tests/vllm/models/test_kolibri1_decode_bench.cpp`, target
`test_kolibri1_decode_bench`)

Shape: one prefill over a deterministic 128-token prompt
(`(i*7919+13) % 128000`), then 63 incremental t=1 greedy decode steps
through one persistent paged KV topology — 64 greedy tokens out. Greedy,
single request, bf16 KV, sliding-window 513 never binds at this length.

| axis | 8 threads | 4 threads |
|---|---|---|
| wall (load excluded) | 62.7 s | 120.4 s |
| prefill (128 tok) | 13.0 s | 24.9 s |
| decode (63 tok) | 49.7 s | 95.5 s |
| decode tok/s | 1.267 | 0.660 |

The 8-thread numbers reproduce the scratch recipe's recorded baseline
(49.39 s decode, 1.276 tok/s, 12.67 s prefill — the negative-record table)
within run noise. The chained greedy sequence alternates
`101807, 109726, 101807, 109726, …`; the prefill's first token is 101807 and
the **pinned last-token anchor is 109726**, captured on main 323f81ca6.

## Determinism and the process-state sensitivity

The negative record's §Diagnosis 4 reported an uncached pass producing a
different chain after a cached pass in the same process. This unit could not
recover that harness (the lever was reverted; the header no longer exists on
any branch here), so the cached-then-uncached leg remains open under
ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2. What this unit measured is the
cache-free sensitivity, and it did NOT reproduce: on pristine main, the
64-token greedy chain is byte-identical across

- separate processes (3 runs),
- `VLLM_CPP_CPU_THREADS=8` vs `=4`,
- two back-to-back passes in ONE process (fresh engine and topology per
  pass, shared process state).

So the sensitivity, whatever it was, either needed the cache resident or was
an artifact of the reverted scratch harness; the committed bench therefore
does not need cross-pass isolation to be a stable gate (ctest isolates per
binary anyway, and the bench is one TEST_CASE). The committed gate asserts
(i) completion of all 64 tokens and (ii) the anchored last token 109726;
timing is a non-asserting MESSAGE.

## Method

`ctest -R test_kolibri1_decode_bench` with `VLLM_CPP_CPU_THREADS` set; the
bench excludes weight loading from the timers, times prefill and decode
separately (`std::chrono::steady_clock`), and reports wall / tok/s as
MESSAGE. Contention: box otherwise idle; the 4-thread run is the same binary,
only the thread count differs.
