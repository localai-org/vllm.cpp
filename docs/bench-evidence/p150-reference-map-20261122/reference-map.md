# P150 benchmark-reference map — Lottolabs Inkling-2 vs vllm.cpp DFlash2

Row: `QWEN38-DFLASH2-STAGE-C`. Issue: ISSUE-LOCAL-01M82GS62HY2S9FT8R69JYR8CQ.
Date: 2026-11-22. Status: **analysis only; no source change, no benchmark run**.
The purpose is to define honest comparison shapes before a P150 benchmark row is
claimed, not to publish a parity result.

## Sources

- Lottolabs package: `/mnt/models/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150`, HF
  revision `b0ad7060246f192aa1823b6fb8bd17f6ead194ae`, downloaded and verified
  with `sha256sum -c SHA256SUMS` (482/482 OK, zero failures). Referenced files
  are `README.md`, `runtime/inkling2/tt/kvplan.py`,
  `runtime/inkling2/tt/NOTES-{model,spec,attention,longctx}.md`,
  `runtime/inkling2/tt/fast/NOTES-{attn,prefill}.md`, `evidence/lmx/summary.md`,
  `native/gptq/native_manifest.json` and `checkpoint/config.json`.
- Our state: W28/W30/W31 evidence in
  `../qwen38-dflash2-stage-c-{w28,w30,w31}-*/README.md`,
  `.agents/specs/dflash2-spec-decode.md`,
  `.agents/specs/tenstorrent-bfp-weight-residency.md`, and `docs/ENVIRONMENT.md`.

**Caveat that governs every number below.** The two sides are different models,
different quantizers and different runtimes. Lottolabs measures Inkling-2-Mini
(26.4B / 3.4B active, 24 layers x 3 passes over 6-17, 128-expert top-8 MoE) with
GPTQ-BFP4 native weights. Our W30 numbers are Qwen3.8-27B Q4_K_M with the
in-engine BFP8 DNR path. No number in this note is a parity claim; each row of
the final table names the unit that could produce one.

## 1. Decode buckets — what a fair trace-replay comparison must replicate

### Their scheme

`tt/kvplan.py` defines `MIN_BUCKET = 4096`, `FAST_MAX = 65536`,
`LONG_MAX = 131072`, `NATIVE_MAX = 1048576`, and `decode_buckets(max_seq_len)`
returns powers of two from 4096 to the profile bound. `bucket(pos)` selects the
smallest full length greater than the position, and `set_decode_pos` /
`decode_step` / `decode_traced` switch buckets transparently with shared caches.
Per bucket there is one trace and one `AttnDecodeTables`; all bucket tables and
the trace input are allocated in the constructor. `TRACE_PER_BUCKET = 72 MiB`
(measured, rounded up) plus `TRACE_FIXED = 64 MiB`, with a 512 MiB trace region
for six buckets at 131072. Their notes give the reason directly: a single trace
sized for the maximum context costs +26 ms per step.

Measured traced step cost by bucket (NOTES-model.md, one token, includes
readback): 4096 -> 68.4 ms, 8192 -> 69.1, 16384 -> 70.7, 32768 -> 74.1,
65536 -> 80.7, 131072 -> 94.1. Decode profiles tie to KV format: `fast` uses
BF16 K/V up to 65536, `long` BFP8/BFP8 to 131072, `native` BFP8/BFP4 to
1048576. Prefill chunks are capped at 4096 (MoE route program L1 limit) with a
512-token tail schedule.

### What a vllm.cpp port must hold constant

A trace-replay comparison is only meaningful if both sides use:

1. the same bucket boundaries and the same position-to-bucket rule;
2. per-bucket tables and trace input allocated **before** the first capture, so
   no post-capture allocation can land in freed trace intermediates (this is
   the mechanism behind their second-sequence hang, NOTES-model.md lines
   97-124);
3. all bucket programs compiled before capture, because their decode programs
   own reshape maps too;
4. the same KV dtype profile at the same context, since BFP8/BFP4 tiers change
   both latency and KL (their native profile raises all KL 0.0112 -> 0.0143);
5. the same prefill chunk schedule and warmup order; and
6. an explicit label of **eager vs traced**. Our W28 struck leg was fully eager
   (`capture_hit = 0`) and still produced the writer's 512-byte signature, so
   mixing an eager leg with a traced Lottolabs number would compare different
   execution paths.

Our engine has no equivalent bucket scheme today, and the W30 gate uses TPOT
over a 12-token output rather than a per-bucket traced step. A P150 port would
first need a bucket-aware traced decode path before this axis can be compared.

## 2. Device-side exact sampling and adaptive speculative decoding

### Their design

The drafter is the upstream 3-layer block drafter (`checkpoint/mtp.safetensors`,
317.9M BF16 params): target features at layer visits 5 / 23 / 41 plus the
final-norm output, a rank-256 recurrent Markov head, an 8-row DFlash
fill-in block with non-causal block attention, and argmax draft tokens. Verify
uses the target's rows for the anchor plus drafts, accepts the longest prefix.
`Spec5` added adaptive R (1-3 drafts per cycle), a pipelined cycle and a BFP4
draft base; sampled requests use **speculative rejection sampling** (accept the
draft with probability p(draft), otherwise sample the residual), which preserves
the distribution exactly.

Sampling runs on device over the full vocabulary for temperature, top_k unset
or 1-64, top_p, min_p, penalties and seeds. Host fallback is limited to
logprobs, logit bias, structured output, temperature > 8, and top_k > 64 with
top_p < 1. Greedy speculative and plain output are bitwise identical on their
final runtime; seeded sampled spec and plain produce the same tokens.

Published reference (README.md, LocalMaxxing matrix, single sequence, dev-18):
greedy plain 87.1 tok/s mean, greedy spec 108.0 (1.24x), sampled T=1.0 spec
98.5 (1.16x), T=0.7 spec 100.9 (1.18x); context 0 -> 128K aggregate decode
82.8 -> 58.4 tok/s spec on, 82.8 -> 60.4 spec off. Their spec on equals spec off
on the prose bench because drafts are accepted too rarely there.

### Our DFlash2 state and the honest comparison shape

Our DFlash2 acceptance has been measured at ~2.49 accepted draft tokens/step
against vLLM's ~3.13 (~20% fewer), attributed by D6/D7/D8 to bf16-irreducible
near-tie divergence, per-step context-KV recompute, and eager-vs-graphed draft
launch. The W30 gate runs explicit drafter-ON and drafter-OFF arms on the same
binary: with the DNR repair armed, drafter-ON passes its band at 22/24 tokens
equal to the recorded `d49c53d2` with one 3-token near-tie shift (a 500-mnat
tie-band gate passes it, the corruption-class gate does not); drafter-OFF is
byte-identical 6/6. The DNR-off proof leg still fails because the writer is
unfixed, and the measured acceptance multiple from the repair is ~3.73x
(24794.79 ms vs 6647.67 ms TPOT).

An honest comparison therefore requires: same workload and prompt set, same
model and quantizer, same drafter family (their upstream Inkling-2 block drafter
vs our DFlash2), same context profile, batch 1, temperature/top_k fixed, the
same acceptance metric (accepted draft tokens per step) reported alongside
tok/s, both drafter arms, and both DNR arms on our side until W32 closes the
writer. Reporting their 1.24x spec speedup next to our 3.73x repair multiple as
if they were the same axis would be misleading: one is a spec gain, the other
the cost of a mitigation for an unfixed defect. Their greedy spec == plain
bitwise property is also stronger than ours near ties, so token identity must
be its own row, not folded into throughput.

## 3. Native BFP4 package vs `VT_TT_WEIGHT_RESIDENCY=bfp8`

They are **the same payload dtype family but a different residency and
provenance class**.

Their `native/gptq/` is an offline-built, hardware-specific artifact: format
`inkling2-mini-tt-v2`, 16.0 GB, routed experts of all 24 layers as GPTQ-quantized
BFP4 device tile binaries plus BFP4-exact attention projections stored as
`bf16` W^T (attn/), with per-tensor format/shape/hash manifest, load contract,
toolchain pin (image, tt-metal commit), GPTQ calibration provenance and a
bitwise full-logits reload proof (`equivalence.json`: prefill [96, 200058] and
decode [17, 200058] logits and tokens bitwise equal). 12.81 GiB of experts on
device; the embedding table is kept on the host to save 0.86 GiB. The quality
arm uses BFP8 from the checkpoint's NVFP4/MXFP8 (24.20 GiB).

Our `VT_TT_WEIGHT_RESIDENCY=bfp8` is a runtime transcode at the staging seam:
`off` (default) keeps byte-identical bf16 staging; `bfp8` uploads bf16 and
typecasts to `BFLOAT8_B` on first staging, roughly halving resident bytes;
`bfp4` is reserved and refused by name. `EnsureBfp8WeightDevice` in
`src/vt/tenstorrent/tenstorrent_residency.cpp:876` is the seam: H2D `from_span`
temp at ~:906, `ttnn::typecast(..., BFLOAT8_B)` at ~:914. It is not a prebuilt
quantized artifact: there is no offline calibration, no per-tensor manifest of
the staged copy, and its bytes are a deterministic conversion of the source
weights rather than GPTQ codes snapped into BFP4 geometry. The W28 writer
strikes exactly these freshly staged BFP8 buffers, which is why our residency
axis currently carries a repair cost their prebuilt package does not.

Consequences for comparison: quality KL is not comparable (their numbers
reflect GPTQ calibration; ours reflect plain BF16-to-BFP8 conversion); memory
footprint is not comparable without stating expert dtype and embedding
placement; and the correctness axis must include the staging-writer state,
which their package never exercises.

## 4. Benchmark axis map

| Axis | Our current state (Qwen3.8-27B / DFlash2 / P150) | Lottolabs reference (Inkling-2-Mini / GPTQ-BFP4 / P150) | Unit that would produce the comparison |
|---|---|---|---|
| Traced decode step, 4K bucket | Not recorded per bucket; W30 uses TPOT over 12-token legs; untraced step ~6.65 s with DNR off | 68.4 ms traced at 4096, rising to 94.1 at 131072 | New P150 benchmark row: identical prompt, bucket and trace recipe, both arms |
| Decode vs context ladder | Q4_K_M KV64 and UD-Q6_K_XL KV8 arms; Q6_K_XL drafter-ON capacity-blocked | 0 -> 128K: 82.8 -> 58.4 tok/s (spec on), 82.8 -> 60.4 (off) | Same lmx matrix recipe on a ported model, ctx 0/4K/16K/32K/64K/128K, batch 1 |
| TTFT / prefill | Not measured on P150 for this model | 0.119 s at 128 tokens, 9.123 s at 32768; prefill 1080 -> 3592 tok/s | P150 prefill row with same chunk schedule and KV profile |
| Speculative speedup | DFlash2 acceptance ~2.49 vs vLLM 3.13; near-tie divergence; acceptance multiple ~3.73x from unfixed writer | greedy 87.1 -> 108.0 tok/s (1.24x); sampled T=1.0 1.16x | Spec row reporting acceptance tokens/step and tok/s separately, both drafter and both DNR arms |
| Token identity spec vs plain | DNR-armed 22/24 tokens vs golden with 3-token near-tie shift; drafter-OFF 6/6 byte identical | greedy spec == plain bitwise; seeded sampled spec == plain | Correctness gate on same golden, reporting near-tie margins |
| Sampled decoding exactness | Not measured on device over full vocab | device full-vocab exact sampling; host fallback list | Sampling parity row with same seed/top_k/top_p grid |
| Long context | Not served at 1M | 1,048,000-token prompt, 3/3 pass keys, 23.0 tok/s decode | Native-context row with same passkey procedure |
| Weight residency memory | `bfp8` runtime transcode, ~half of bf16; DNR repair cost while writer live | 12.81 GiB GPTQ-BFP4 experts, embedding on host, 16.0 GB package | Residency row stating expert dtype, calibration and embedding placement |
| Quality vs BF16 | BF16-to-BFP8 conversion only; no GPTQ calibration | default all KL 0.0112 / top-1 0.966; `--precision quality` 0.0030 / 0.981 | Quality row on identical eval set before any KL comparison |
| Suite-state correctness | W28/W31 writer live, DNR default-on; RAC and PA latent issues open | no equivalent defect published; trace-safety fixes documented | W32 dispatch trace + the two falsification experiments in PR #3592 |

## Owed before any comparison is published

1. W32 must resolve the writer mechanism; until then our decode and spec
   numbers carry a mitigation cost and are not comparable to Lottolabs'.
2. A P150 benchmark row must pin the HF revision above, the model hash, the
   exact recipe, an idle-device `rc` lease, and report both drafter arms on
   the same binary.
3. A port would need either the native package's manifest-equivalent for our
   weights, or an explicit statement that the comparison is runtime-transcode
   to runtime-transcode.
