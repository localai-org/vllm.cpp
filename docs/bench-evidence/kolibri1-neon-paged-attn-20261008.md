# NEON paged-attention lane — kolibri-1 evidence, 2026-10-08

Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm. Issue:
`.agents/issues/MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm/ISSUE-LOCAL-01M4EQ7TPR5Q7JQGNM8628X9HW.md`.
Branch `row/kolibri-neon-attn` (base bf654ae...bf7b654ae origin/main, verified with
`git log -1`). All numbers on this host (aarch64, 128 cores, 4 NUMA nodes),
`numactl --interleave=all`, model `/mnt/models/Aleph-Alpha/Kolibri-1`.

## Variant matrix covered

The scalar kernel (`src/vt/cpu/cpu_paged_attn.cpp`) serves: query f32/f16/bf16;
KV cache f32/f16/bf16 and fp8-e4m3 (KV-FP8 W1, dequantized by k_scale/v_scale);
out f32/bf16; GQA (qpk = hq/hkv); causal and non-causal; sliding window
(AttentionWindow left/right); logits soft-cap; per-head attention sink
(MiMoV2); MiMoV2 v_head_dim (d_v != d); paged block table with the #1394
short-table refusal. The NEON lane serves the SAME set: it only replaces the
K dot-product and the V accumulation with float32x4 fma over the same element
converters, gated per head on `d % 4 == 0 && d_v % 4 == 0`; everything else
(masking, softmax passes, sink seeding, GQA mapping, paging) is the untouched
scalar code. Any width not a multiple of 4, any non-aarch64 build, and any
build with `VT_CPU_PAGED_ATTN_NEON` unset or `=0` runs the scalar body.

## Envelope decision

NOT bit-exact, stated before the first run (test header,
`tests/vt/test_ops_paged_attn_neon.cpp`): the K reduction becomes four
interleaved fma lanes summed by `vaddvq_f32`, so scores drift by f32
accumulation noise and outputs are adjudicated at rel 2e-5 / abs 2e-6 with one
bf16 output ulp (2^-7 relative) allowed for rounding-boundary straddles.
Whether the drift is acceptable in production is adjudicated by the model
gates, not by the memcmp-style test.

## Red-first and mutation evidence

- First test run (envelope v1, abs 2e-6 only) was RED: one element at
  `decode t=2 kv=f16 q=bf16 elem 1285` straddled a bf16 rounding boundary
  (0.000648 vs 0.000645) — the envelope was widened to one bf16 ulp, then
  green. The failure was a tolerance artifact, not a kernel defect.
- REACH anchor (aggregate): the raw NEON output must bit-differ from scalar in
  more than half the swept configurations. Mutation (lane selector wired to
  constant false, scratch build): anchor at **0 of 272**, REQUIRE red; restored
  byte-for-byte, green at 190/272.

## Unit sweep

`ctest -R test_ops_paged_attn_neon` — T in {1,2,8,64}; GQA {4/4, 8/4, 8/1,
32/4} at dh 128 and 64; sliding window {16,0}; non-causal full; softcap;
block_size 16 spanning; varlen with an empty row; the kolibri production shape
(GQA 48/4, dh 128, T=128); KV dtype {f32, f16, bf16, fp8 (0.75/1.25 scales)}
x query {f32, bf16} x out {f32, bf16}; 2.28M -> 14.86M assertions after the
kolibri case was added, all green.

## Replay of the real call

Scratch hook (`VT_DEBUG_DUMP_ATTN`) dumped the first real kolibri prefill
call (bf16 query/cache, d=d_v=128, hq=48, hkv=4, bs=16, T=128, no fp8, no
softcap, no sink). Replaying those exact bytes through both paths: NEON vs
scalar maxdiff **0.00195312 (1 bf16 output ulp), 0 elements out of envelope**.
The kernel is numerically correct on the real operands.

## A/B — kolibri decode bench (64 greedy tokens, 128-token prompt)

`test_kolibri1_decode_bench`, `VT_KOLIBRI1_PROFILE=1`, 3 reps each, loadavg
recorded per rep (host was under concurrent load from other agents; treat the
tok/s deltas as indicative, not final):

| threads | knob | tok/s (3 reps) | attn_core s / 64-step run |
|---|---|---|---|
| 32 | off | 2.85 / 2.96 / 3.24 | 4.85–4.91 |
| 32 | on | 3.66 / 3.67 / 3.73 | 1.88–2.32 |
| 8 | off | 1.265 / 1.272 / 1.265 | 5.29–5.31 |
| 8 | on | 1.332 / 1.356 / 1.351 | 2.52–2.54 |

attn_core halves (≈2.1x) at both thread counts. Wall moves less because
attn_core is ~10% of the decode wall at this shape (the 2026-10-04 profile
ranking holds: moe_glue dominates): +6% tok/s at t8, ~+15–25% at t32 under
load noise. An idle-host re-measurement is owed before any default flip.

## Cross-model regression

- kolibri battery, NEON ON: test_kolibri1 234/234; test_kolibri1_dequant 10/10;
  test_kolibri1_dequant_cache 52/52; test_kolibri1_moe_glue 21/21;
  test_kolibri1_w2 1608/1608. test_kolibri1_tt / _tt_b2i / _tt_b2bi are
  Tenstorrent-build gates (not registered in a CPU-only build; device legs, out
  of scope for this CPU seam change).
- **W3, NEON ON (t8, quiet-checked): 900/900 assertions, ARGMAX CHAIN 141/145,
  4 flips ALL near-tie, 0 hard flips.** Worst topk logit diff 2.05711 vs the
  committed scalar fingerprint 2.18646 — the NEON arm's drift is SMALLER than
  the scalar kernel's own drift against the reference goldens. Fingerprints
  moved (worst sum diff 29045.1 vs 26763) inside the 2.5-nat band, as the
  near-tie contract anticipates.
- test_qwen3_paged_engine: FAILS IDENTICALLY WITH THE LANE OFF (same anchor
  drift prompt[0] tok=5, engine=15344 vs committed 9625) — a pre-existing
  host drift at origin/main behavior on this aarch64 box, not a NEON
  regression. With the lane ON the drift positions are unchanged or adjacent
  (prompt[4] tok=10 on 4B) and the in-test vLLM near-tie gap checks still pass.
- test_qwen35_paged_engine: skip-by-design on CPU (ROCm-oracle-captured
  goldens; exit 77) in both knob states.

## The decode-bench strict anchor — adjudication and the flip decision

`test_kolibri1_decode_bench` pins the greedy chain BYTE-EXACTLY
(anchor 109726, alternating 101807/109726). With the lane ON, the first
decoded token flips (NEON argmax 33382 vs scalar 101807) and the chain
degenerates, deterministically at both t8 and t32. Adjudication:

- The scalar top1-top2 gap at that position is **0.197 nats** — inside the
  2.5-nat near-tie band the row already uses.
- The replay of the exact real attention call shows the NEON path agrees with
  scalar to 1 bf16 ulp; the flip is amplified bf16 rounding under a
  strict-identity gate, the same class W3 adjudicates (and W3, which IS the
  near-tie contract, passes with 0 hard flips).

Disposition: **the default stays OFF.** `VT_CPU_PAGED_ATTN_NEON=1` is the
opt-in; the scalar path remains the default and the reference. Flipping the
default is an operator decision that owns two things: (1) an idle-host A/B
re-measurement, and (2) a near-tie adjudication for the decode-bench anchor
(the bench's token-identity gate predates the near-tie band and has no gap
instrument; relaxing it is a gate-semantics change needing its own spec).
Never weakening the gate to make the lane green: the gate stands; the lane
simply is not the default until the gate question is settled.

## Reproduce

```sh
cmake /tmp/vllm-kolibri-neon-attn -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF -G Ninja
ninja test_ops_paged_attn_neon && ctest -R test_ops_paged_attn_neon
VLLM_CPP_CPU_THREADS=8 VT_CPU_PAGED_ATTN_NEON=1 numactl --interleave=all \
  ./tests/test_kolibri1_decode_bench     # tok/s + attn_core via VT_KOLIBRI1_PROFILE=1
VLLM_CPP_CPU_THREADS=8 VT_CPU_PAGED_ATTN_NEON=1 ./tests/test_kolibri1_w3
```
