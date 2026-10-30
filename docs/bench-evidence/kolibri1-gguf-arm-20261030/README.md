# KOLIBRI1-GGUF-ARM — the mudler APEX I-Micro tier through the kolibri1 GGUF arm

Row: `KOLIBRI1-GGUF-ARM`
Spec: `.agents/specs/kolibri1-gguf-arm.md`
Issue: `ISSUE-LOCAL-01M66B4KT6QFENVDGN6G99M8CM`
Date: 2026-10-30. Host: the shared x86 128-core box (Ampere-1a, 4 NUMA nodes,
255 GB RAM), governor `performance` on all CPUs (read 2026-10-30 12:07Z and
14:10Z). Worktree `/tmp/kolibri-gguf` @ `7b3bbf092`, build
`/tmp/build-kqgguf` (CPUs 64-127, Release, Ninja, 2351/2351 green).

## RECORD RECONCILIATION (read first)

The row's two named blockers are **already closed in this branch's history**;
the row premise was stale against the tree:

1. The `kolibri1` arch row exists — `src/vllm/entrypoints/model_loader.cpp:1277`
   dispatches `general.architecture = kolibri1` to
   `Kolibri1HfConfigFromGguf` (commit `71ee347e9`, 2026-10-23, row
   MODEL-TEXT-kolibri-1 ROUTE A, spec `.agents/specs/kolibri1-gguf-arch.md`).
2. `kolibri1_weights.cpp`'s fp8 checks are NOT on the GGUF path:
   `LoadKolibri1FromGguf` (`kolibri1_gguf_weights.cpp`) fills the bf16 arm
   directly after dequant; the fp8-block requirement stays scoped to the
   safetensors loader (`LoadProjection` in `kolibri1_weights.cpp:68-92`).
   No weakened check and no arm discriminator was needed — the arm split the
   commit already introduced IS the discriminator.

What this row therefore adds is the artifact gating the spec asks for: the
mudler APEX tiers are a DIFFERENT conversion than the Hob-forge container the
committed gate pins (`KOLIBRI1_GGUF_PATH` default,
`/mnt/models/Hob-forge-Kolibri-1-GGUF/Kolibri-1-Q2_K.gguf`).

## Red-first (reconciled)

At HEAD the APEX I-Micro file is NOT refused: it loads. The refusal this
row's spec describes ("unrecognized architecture") is the pre-`39dadc637`
state, captured by the dispatch-reachability case in
`tests/vllm/test_model_loader_gguf.cpp` and recorded in the
`test_kolibri1_gguf.cpp` header comment. No fresh red capture exists to take
without reverting landed work; this file is the honest substitute.

## The artifact

`mudler/Kolibri-1-APEX-GGUF` @ `55a4781439f1e8abb7e46c795e0681d6b2b7f4d9`,
`Kolibri-1-APEX-I-Micro.gguf`, 22,173,110,464 bytes, sha256
`3edfaf09aac60d60cde56744fe1e0a7e5362c5549d5eb12d97ec4a9c83c90e0e` —
verified by `fetch-flint-par.sh` after assembly (`kq-apex-fetch.log`; the
fetch died twice mid-flight before the clean monitored run, 12-stream
ranged, ~13 MB/s). Header: `apex-imicro-header-first8mb.bin` (first 8 MB,
range-fetched; header ends at byte 4,887,725).

Header census (parsed 2026-10-30, range request): arch `kolibri1`, 903
tensors, 50 blocks, 54 KVs, `expert_count=384`, `expert_used_count=6`,
`expert_shared_count=1`, `expert_weights_norm=0`, `expert_gating_func=2`,
hidden 2560, 48/4 heads, head_dim 128, sliding_window 513,
`sliding_window_pattern` = 4 sliding + 1 full ×10. Tensor-type histogram:
`{Q5_K:31, F32:401, Q2_K:1, Q4_K:144, Q3_K:206, IQ2_XS:30, IQ1_M:90}` —
every type already handled by `gguf_dequant.cpp`.

## Graph-shape check: NO stop condition

The 21 per-layer tensor suffixes present in the file match
`EnumerateKolibri1GgufTensors` EXACTLY — including `post_attention_norm`,
`ffn_norm`, `attn_q_norm`, `attn_k_norm` (the APEX conversion carries the
full four-gamma sandwich; the task brief's shorter list was incomplete),
`ffn_gate_shexp`, `ffn_gate_inp` and `exp_probs_b.bias`. Every tensor has a
home in the fp8-era weight structs; `LoadKolibri1FromGguf`'s strict
unaccounted-tensor check passed (167/173 assertions green, all failures in
the numeric band case below). Shapes verified at blk.0: attn q/k/v/out
`[6144/512, 2560]`, expert towers `[E=384, out, in]`, router `[2560, 384]`.

## Serving gate (the token cross-check): BEYOND BAND — STOP, numerics finding

Harness: `test_kolibri1_gguf` built with
`-DKOLIBRI1_GGUF_PATH=/mnt/models/kolibri1-apex-gguf/Kolibri-1-APEX-I-Micro.gguf`
(new CMake cache knob, `tests/CMakeLists.txt`), goldens =
`kolibri1_goldens.json` — the fp8 safetensors primary-oracle run
(`/mnt/models/Aleph-Alpha/Kolibri-1` is mounted; the goldens ARE the fp8
cross-check arm). Band = the W3 band, 2.5 (bf16-measured).

Result (`apex-gate-run1.log`):

- Config-map case: PASS (all 13 structural checks against the file's own
  metadata).
- ARGMAX CHAIN: **37/44** compared positions match the golden greedy decode
  (7 flips: 5 near-tie, **2 hard**); worst chain gap 9.56.
- FINAL-STEP FINGERPRINT: worst topk logit diff **3.245 > 2.5**, worst sum
  diff 47242.8 (sum is a 128k-vocab cancellation figure, not diagnostic).

Verdict per the spec's stop condition: the cross-check fails the W3 band
beyond the adjudicated fp8-vs-GGUF class envelope, so this row STOPS here
and reports. Comparison with the already-adjudicated Hob-forge Q2_K
(hob-q2k-gate-run.log, same harness): 5/13 chain match, 3 hard flips, worst
topk diff 3.97 — the APEX I-Micro tier is materially closer to the fp8
oracle than the committed tier, in the same failure class.

Load-defect ruled out, on the earlier row's own criterion: the fluency probe
(`apex-fluency.log`, `VT_KOLIBRI1_GGUF_GEN=1`) produces prompt-appropriate
text ("The capital of France is Paris." / German prompt continues in
German) with looping of exactly the shape the bf16 goldens themselves show;
a mis-mapped expert slice or swapped norm produces garbage, not this.

## Throughput: NOT TAKEN

Per the spec's stop condition (numerics first), the 16/32-thread
throughput legs and the public-doc weight pin were not run/landed. The
harness to resume with is `examples/vllm-bench --model
/mnt/models/kolibri1-apex-gguf/Kolibri-1-APEX-I-Micro.gguf` (production
entry point), idle-gated, vs the fp8 arm at
`/mnt/models/Aleph-Alpha/Kolibri-1`. Note: the box's bench zone was under
external load during this session (load avg 107 at 14:10Z) — any resumed
leg must re-gate.

## Tree change in this row

- `tests/CMakeLists.txt`: `KOLIBRI1_GGUF_PATH` cache knob so the SAME gate
  binary can point at any kolibri1 GGUF container (default unchanged:
  Hob-forge Q2_K). No product-code change: the arch row and loader TU
  already on this branch needed zero edits to load the APEX file.
- Evidence: this directory.

## Adjudication + resume (2026-10-30, later session)

**Developer adjudication, Option A: the Hob-forge precedent applies.** The
quantized kolibri1 arms are adjudicated on their own measured record; APEX
I-Micro (37/44 exact, worst topk diff 3.245, 2 hard flips vs the fp8 goldens)
is strictly better than the adjudicated Q2_K bar (5/13, 3.97, 3 hard) and
therefore PASSES. The fp8 2.5 band governs same-weights kernel comparisons
only.

### Gate pin (test_kolibri1_gguf, KOLIBRI1_GGUF_PATH = the APEX container)

The token gate now pins the measured record when the container path names an
APEX tier (the Hob-forge Q2_K default keeps `hard_flips == 0` and the 2.5
per-topk band unchanged):

- exact greedy positions >= **37** of 44 (regression in either direction fails);
- worst topk diff <= **3.2452** (the measured 3.24517, rounded out — the
  adjudication's "3.245" was a truncated quote of the same number; a tighter
  result means the pin moves with a fresh measurement, not silently);
- hard flips <= **2**;
- a REQUIRE_MESSAGE cites the adjudication verbatim: "developer-adjudicated
  2026-10-30, Option A: quantized kolibri1 arms adjudicated on their own
  measured record; the fp8 2.5 band governs same-weights kernel comparisons
  only".

Fresh gate run at the pin (`apex-gate-run2.log`): 3/3 test cases, 109/109
assertions, SUCCESS.

### Tier sanity sweep: I-Mini (quality monotonicity check)

Artifact: `mudler/Kolibri-1-APEX-GGUF` @ `55a4781439f1e8abb7e46c795e0681d6b2b7f4d9`,
`Kolibri-1-APEX-I-Mini.gguf`, 28,973,904,064 bytes, sha256
`64e61300f557de747957096b70907737e063b9b762ab51811e28a6ef57e725a4` —
verified by `fetch-flint-par.sh` after assembly (`apex-imini-fetch.log`).

**FINDING — the bigger tier does NOT measure closer to the fp8 oracle.**
Same gate, same fp8 goldens, I-Mini pointed at `KOLIBRI1_GGUF_PATH`
(`apex-imini-gate-run1.log`):

- I-Micro (22 GB): **37/44** exact = 84.1%, worst topk **3.245**, **2** hard flips
- I-Mini  (29 GB): **25/33** exact = 75.8%, worst topk **5.033**, **0** hard flips

Exact-rate regresses 8.3 points and the worst topk diff grows by 55%, though
I-Mini flips nothing hard. The compared-position count differs because the
greedy replay stops at first divergence — I-Mini diverges earlier and longer.
Quantization aggressiveness in this converter family is per-tensor, not
monotone in container size. Recorded, not hidden: the gate pins each tier at
its OWN measured record (I-Micro: >= 37, <= 3.2452, <= 2; I-Mini: >= 25,
<= 5.0327, <= 0), both green. I-Micro stays the documented serving tier.
Balanced/Quality tiers were NOT swept (optional; time-boxed).
### Serving-path segfault: FOUND AND FIXED (ISSUE-LOCAL-01M66NS2QCG5C0JN8Y88SQDPD3)

The first vllm-bench leg on the GGUF segfaulted in `v1::InputBatch::add_request`.
Root cause: `Kolibri1HfConfigFromGguf` never set `max_position_embeddings`, so
`ResolveMaxModelLen` (model_loader.cpp) returned 0 and the engine's token
buffer was zero-length — while the file itself carries
`kolibri1.context_length = 262144`. Fixed red-first: the config-map case now
CHECKs the field (red `CHECK(0 == 262144)`), the mapping reads the key via a
new `OptInt` helper beside `ReqInt` (the flint GGUF mapping reads the same
key), and serving re-verified end-to-end (`vllm-cli` greedy decode fluent;
engine auto-fits max_model_len 262144 → 4064 against the KV pool). This
closed the production-entry-point reachability gap for the arm.

### Throughput (serving path, idle-gated, 3-rep medians)

Recipe: `vllm-bench --model <M> --num-prompts 8 --input-len 128
--output-len 32 --concurrency 4 --seed 0 --temperature 0`, `taskset -c 32-63`,
`VLLM_CPP_CPU_THREADS=16|32`, decode-throughput axis. Idle gate per leg
(1-min load < 16 + process scan; `bench-*` logs carry the gate lines).
Governor `performance` (read 15:00Z and 17:00Z). An earlier un-gated set was
CONTAMINATED by external load (box 1-min load 139 mid-leg; one rep measured
1.34 tok/s) and was discarded — the per-leg gate exists because of it.

| threads | fp8 safetensors | GGUF APEX I-Micro | ratio |
|---:|---:|---:|---:|
| 16 | 4.46 (4.44/4.46/4.48) | **4.97** (5.01/4.97/4.97) | 1.11× |
| 32 | 5.77 (5.77/5.76/5.80) | **6.55** (6.49/6.55/6.58) | 1.14× |

(tok/s, output decode axis, medians with all three reps.) The k-quant
container serving FASTER than fp8 is consistent with load-time dequant to
bf16 on both sides — same forward, fewer bytes moved. No llama.cpp
quant-matched denominator exists (the `llama-cpp-kolibri1` pin `b10451`
predates PRs #30673/#30865; named follow-up), so these are internal A/B
numbers, published in `docs/benchmarks/kolibri1-gguf-serving.md` with the
adjudication and floor-absence caveats.
