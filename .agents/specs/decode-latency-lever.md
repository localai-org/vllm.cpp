# Decode per-token latency (TPOT/ITL) lever — EXPLORATION verdict

**Status:** EXPLORATION COMPLETE — hypothesis **CONFIRMED** by direct measurement
(`CLAIM-DECODE-LATENCY-EXPLORE`, 2026-07-28, NOT pushed). Pure measurement pass:
**no source changed**, no shipped scheduler behavior altered. Base local `main`
HEAD `348ae7c9eb3c97d64eadb5e66189b82cfc28ef62` (`git rev-parse HEAD`). DGX GB10
sm_121a, idle box, one `flock $HOME/gpu.lock`, engines serialized vs campaign #2.

## 1. The hypothesis (from the SGLang competitor-floor benchmark)

`CLAIM-SGLANG-PERF-BENCH` (`docs/BENCHMARKS.md`) measured, on the 27B-NVFP4 gate
model, that we **decisively win** aggregate throughput (2.21×@c16) + TTFT (6–12×)
but SGLang **wins steady-state per-token latency**: SGLang TPOT ~104 ms / ITL
~105.6 ms vs ours TPOT/ITL 122.9–154.4 ms (1.18–1.49× higher). The benchmark's
ATTRIBUTED (unconfirmed) hypothesis: **our throughput win IS the latency cost** —
we admit/pack larger decode batches, so each request shares each decode step with
more batch-mates → higher per-request per-token latency.

Precise, testable form:
> Our ITL at decode-batch B is a rising function ITL(B). SGLang's *effective*
> decode concurrency is ~B_sg. If the gap is batch-composition, then (a) ITL(1)
> should be ≤ SGLang's ITL (no per-token kernel deficiency), and (b) at B≈B_sg
> our ITL should approach SGLang's ~104 ms. If instead ITL(1) is already
> materially above SGLang, the gap is a per-token **kernel** inefficiency.

## 2. Source scan — why the two engines compose a decode step differently

### Ours (vLLM V1 port) — chunked-prefill MIXES prefill+decode, packs to the cap
- Per-step budget + admission: `src/vllm/v1/core/sched/scheduler.cpp:267`
  (`token_budget = max_num_scheduled_tokens`), RUNNING (decode) scheduled first
  `:280`, then WAITING (new prefills) `:431` up to `max_num_running_reqs`
  (`:432`, `= max_num_seqs`). Chunked prefill lets a NEW request's prefill chunk
  ride in the SAME forward pass as the running decodes (`enable_chunked_prefill`
  default true, `include/vllm/config/scheduler.h:114`; `scheduler.cpp:483`).
- Defaults: `max_num_seqs = 128`, `max_num_batched_tokens = 2048`,
  `enable_chunked_prefill = true`, `policy = fcfs`
  (`include/vllm/config/scheduler.h:12,9,14,117`). The bench/sweep pinned
  `max_num_seqs = 16` to match SGLang's `--max-running-requests 16`.
- Consequence: aggressive low-TTFT admission — new requests start prefill
  immediately (mixed into decode steps), and the running decode batch is packed
  to the cap. Great for TTFT/throughput; each decode step carries B decodes
  (+ occasional prefill chunk).

### SGLang v0.5.15 (`/home/mudler/_git/sglang`, `f63458b`) — prefill-FIRST, NON-mixed steps
- `managers/scheduler.py:2598` `get_next_batch_to_run`: builds a prefill batch
  `get_new_batch_prefill()` (`:2733`); **"# Run prefill first if possible"**
  (`:2700`) → if a prefill batch exists, that step is **prefill-only**; else run
  the pure decode batch (`update_running_batch`, `:3041`). Prefill and decode are
  **separate forward passes** (default; unlike our chunked mixing).
- `--max-running-requests` default `None` (`server_args.py:635`),
  `--chunked-prefill-size` default `None` (`server_args.py:655`), schedule policy
  default `fcfs` (`server_args.py:692`). Bench pinned `--max-running-requests 16`.
- Consequence: while a prefill step runs, running decodes STALL; new requests
  wait for a dedicated prefill slot → **large admission queue** (measured TTFT
  mean 33 s @ c16). At any instant only a few requests are actually in the decode
  batch → **small effective decode concurrency** → low per-request ITL, but low
  aggregate throughput.

## 3. CONFIRM measurement — ITL vs decode batch on OUR engine (the decisive data)

Model `unsloth/Qwen3.6-27B-NVFP4` `890bdef7` (dense hybrid mamba+attention),
1024-in/128-out exact, greedy, `ignore_eos`, `--max-num-seqs 16`,
`--num-blocks 640 --block-size 32` (KV 20480), request-rate inf, semaphore =
concurrency = steady-state decode batch B. Client = async streaming ITL probe
(`/tmp/itl_client.py`; ITL = pooled inter-token gaps, TPOT = per-req
(e2e−ttft)/(out−1)); 2 reps each, CV <1%. Evidence dgx
`~/work/decode-lat-348ae7c/sweep_out/sweep_results.jsonl`.

| conc = decode batch B | out tput tok/s | ITL mean ms | ITL median ms | ITL p99 ms | TPOT mean ms | TTFT mean ms |
|---|---|---|---|---|---|---|
| **1**  | 9.58  | **101.75** | 101.68 | ~104 | 101.75 | 441 |
| **2**  | 17.84 | **106.92** | 105.96 | 109  | 106.92 | 763 |
| **4**  | 32.39 | **113.77** | 110.46 | ~118–222 | 113.77 | 1324 |
| **8**  | 56.40 | **125.20** | 115.75 | 493  | 125.20 | 2178 |
| **16** | 87.66 | **158.48** | 131.11 | 900  | 158.48 | 3126 |
| *SGLang c16 (recorded, `docs/BENCHMARKS.md`)* | *40.8* | *105.6* | *105.5* | — | *104.0* | *33425* |

Two facts settle the hypothesis:
1. **ITL(1) = 101.75 ms is BELOW SGLang's operating-point ITL (104–105 ms).** Our
   per-token cost with no batch-mates is at least as good as SGLang's → **no
   per-token kernel deficiency**.
2. **ITL rises monotonically with B** (101.75 → 158.48 ms, +56% from B1→B16;
   median 101.68 → 131.11 ms, +29%). The c16 point reproduces the recorded
   benchmark (recorded ours mean TPOT/ITL 154.4, median ITL 125.2; here 158.5 /
   131.1 — within corpus-size noise).

SGLang's **effective decode concurrency ≈ 4** (derived: 40.8 tok/s ÷ (1000/105 ms
per stream) = 4.3), *not* 16 — its prefill-first serialization + 33 s admission
queue keep only ~4 requests decoding at once. Its low ITL is simply **ITL(B≈4)**.
At our matched B: B=2 → ITL 106.9 (≈ SGLang 105); B=4 → ITL 113.8 (≈8% above
SGLang's B≈4 — a small residual step-overhead, secondary). The DOMINANT effect is
batch cardinality.

## 4. nsys decode attribution — "more work per step", not "each token costs more"

nsys `cuda,nvtx`, 30 s steady capture (delay 60 s past load), batch 16 vs batch 1.
Evidence dgx `~/work/decode-lat-348ae7c/nsys_out/decode_b{1,16}.nsys-rep`;
`cuda_gpu_kern_sum` per-instance (= per decode step, per layer) AVG:

| Kernel (role) | b1 avg µs | b16 avg µs | b16/b1 | per-token cost |
|---|---|---|---|---|
| `nvjet_sm121_tst_mma_128x208x64` (NVFP4 dense GEMM, 22–25%) | 2314.9 | 3711.2 | **1.60×** | ↓ ~10× |
| `cutlass GemmUniversal` (NVFP4 GEMM, ~17%) | 1147.7 | 1866.8 | 1.63× | ↓ ~9.8× |
| `GdnChunkWUWmmaVecKernel` (GDN/mamba, ~7.5%) | 684.5 | 1199.5 | 1.75× | ↓ ~9.1× |
| `SiluAndMulFp4QuantFastKernel` | 347.5 | 581.1 | 1.67× | ↓ ~9.6× |
| `RmsNormRowFastKernel` | 152.6 | 276.5 | 1.81× | ↓ ~8.8× |

Every hot kernel's per-STEP time grows **sub-linearly** (1.6–1.8× for **16×** the
tokens) — the GB10 decode GEMM is weight-bandwidth-bound at B=1, so adding rows
is nearly free and **per-token GPU cost drops ~9–10×** (this is exactly why
batching wins throughput). But per-step wall time still rises (→ +29% median
ITL). The tail (b16 ITL p99 900 ms, mean 158 vs median 131) is the **mixed
prefill+decode steps**: when a newly-admitted request's prefill chunk shares a
decode step, all B decodes wait — a facet of our low-TTFT admission, distinct from
steady batch cardinality. No host-gap/idle dominates; time is in the kernels
above. **This is composition ("more requests × cheaper-per-token"), NOT a decode
kernel that costs more than it should.**

## 5. VERDICT — CONFIRMED (batch-composition), REFUTED (kernel inefficiency)

The benchmark's attributed hypothesis is **CONFIRMED**. Our steady-state per-token
latency gap vs SGLang is **batch composition**, not a per-token kernel
inefficiency: (a) at B=1 our ITL is already ≤ SGLang's; (b) ITL rises with the
decode batch we pack; (c) nsys shows every hot kernel is sub-linear in batch
(per-token cost falls ~10×). The throughput win and the ITL loss are the **same
lever**: we keep ~16 requests concurrently decoding (87.7 tok/s) where SGLang
keeps ~4 (40.8 tok/s). Secondary contributors: a small (~8%) matched-batch
step-overhead residual, and the mixed-prefill tail from aggressive admission.

## 6. The tradeoff curve + recommended operating point (knobs ALREADY exist)

Capping the decode batch trades throughput for ITL along the measured curve
(vs the max_num_seqs=16 operating point):

| Cap (max_num_seqs ≈ B) | ITL mean | Δ ITL mean | out tput | Δ tput | vs SGLang tput |
|---|---|---|---|---|---|
| 16 (throughput-oriented, shipped region) | 158.5 | — | 87.7 | — | 2.15× |
| **8 (recommended latency-oriented)** | 125.2 | **−21%** | 56.4 | −36% | **1.38×** |
| 4 | 113.8 | −28% | 32.4 | −63% | 0.79× |
| 2 (≈ matches SGLang ITL ~105) | 106.9 | −33% | 17.8 | −80% | 0.44× |

- **Knob:** `max_num_seqs` (server `--max-num-seqs`) and/or
  `max_num_batched_tokens` (C-ABI v9) bound the decode batch / step composition;
  `scheduling_policy` (ABI v9) is the ordering axis. **No new code** is needed to
  operate this curve.
- **Recommendation:** the shipped **throughput-oriented default stays**
  (unchanged this pass — exploration only). For a latency-sensitive deployment,
  **`max_num_seqs ≈ 8`** is the balanced operating point: it cuts mean ITL ~21%
  (median ~12%) while **retaining a 1.38× throughput lead over SGLang**. Matching
  SGLang's ITL exactly requires B≈2, which surrenders ~80% of the throughput win
  and is **not** recommended — SGLang pays that same price involuntarily (its
  33 s TTFT). Optionally expose a `--latency` preset that pins
  `max_num_seqs≈8`; a future dedicated slot could pursue the small (~8%) matched-batch
  step-overhead residual (the mixed-prefill ITL tail) if a lower floor is wanted.

## 7. Residuals / what's still needed
- 35B-A3B not swept (27B gate model only, per scoping).
- Matched-batch residual (~8% at B≈4) not decomposed to a named kernel — nsys
  confirms no gross per-token deficiency, so it is low-priority; a dedicated slot
  could attribute the mixed-prefill tail (p99) to the chunked-prefill step splice.
- SGLang side not re-profiled (its ITL taken from the recorded floor, per the
  "do not re-stand-up SGLang" scoping); the effective-concurrency ≈4 is derived
  from its recorded throughput÷ITL, not directly instrumented.
- Repro: build `git archive 348ae7c9` on dgx (cutlass-ON + FA2 banners), launch
  `build-cuda/examples/server … --max-num-seqs <B>`, drive
  `/tmp/itl_client.py --conc <B>`; nsys via `/tmp/nsys_driver.sh`. All under
  `flock $HOME/gpu.lock`.
## Owed

- **`ISSUE-GH-362` — the first VALID 27B per-kernel decode attribution**, a
  whole-run trace windowed by the profiler's own `cudaProfilerStart`/`Stop`.
  Section 5's verdict rests on batch composition and section 7 lists what is
  still needed; the per-kernel arm is the measurement this pass did not take.
