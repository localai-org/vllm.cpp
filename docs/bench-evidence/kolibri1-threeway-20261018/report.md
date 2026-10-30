# Kolibri-1 three-way numerics: logit-level distance to the primary oracle (2026-10-18)

Row context: MODEL-TEXT-kolibri-1 (CPU lane). Branch `row/kolibri-three-way`
(fresh from `origin/main` `0c8d22800`).

## Question

Which numerics mode is closest to the PRIMARY oracle (the
aleph-alpha-inference plugin via vLLM, BFMMLA-class) at the LOGIT level, and
is the distance constant or growing with depth?

## Method

The primary published its per-position top-100 logprob rows over the 8-prompt
battery, teacher-forced via `prompt_logprobs=100`
(`tests/vllm/models/kolibri1_primary_topk_rows.json`, byte-identical to
`/tmp/opencode/primary_topk_rows.json`, sha256 `f1a5491da53e64bd…`). The
fixture is NOT dense: the primary dropped positions whose emitted token fell
outside its own top-100, leaving **202 of 256** generated positions scored
(per-prompt scored counts 32/30/24/20/30/20/14/32). Every comparison below is
on those 202 aligned positions.

We built `tests/vllm/models/test_kolibri1_threeway_capture.cpp`: it
teacher-forces OUR CPU forward on the primary's FULL generated chains (prefill
the prompt ids, then feed `generated_ids` one token at a time through one
persistent paged-KV topology) and dumps the full f32 logit row (128,000 vocab)
at each scored position behind `VT_THREEWAY_DUMP_DIR`. Both sides therefore
score position k under the IDENTICAL prefix `prompt + gen[0..k-1]` — the
distance is a numerics distance, not a chain-divergence artifact. Dump dirs
are 99 MB per mode and are not committed; the analysis re-derives everything
from the manifest plus the committed per-position detail
(`threeway_detail.json`).

The analysis script `scripts/kolibri1-threeway-analysis.py` log-softmaxes our
raw logits and computes, per aligned position: the rank of the primary's
emitted token in our full-vocab logprob; the logprob gap `ours − primary` at
that token; cosine similarity over the primary's top-100 support; and
KL(primary‖ours) renormalized on that (truncated) support, since the primary
publishes no tail.

## Modes captured

| mode | tree | knobs |
|---|---|---|
| `prod` | `origin/main` `0c8d22800` (row base) | `VT_CPU_PAGED_ATTN_NEON` unset (scalar body, f32 dequant-fma) |
| `draft` | `mine/row/kolibri-moe-glue-neon` `3a4262bc5` | `VT_CPU_PAGED_ATTN_NEON=1 VT_CPU_KOLIBRI_BFDOT_GEMV=1` |

DEVIATION from the brief: the brief named the FMA-draft knobs
`VT_CPU_KOLIBRI_BFDOT_GEMV` on "the base commit of row/kolibri-fma-draft",
but that branch (`mine/row/kolibri-fma-draft`, base `5b4cbd1e7`) does NOT
carry the BFDOT GEMV arm — `git log --all -S` locates the knob only on
`mine/row/kolibri-moe-glue-neon`, which stacks the FMA-contracted scalar body
(`43ed2a553`), the NEON attention lane, the BFDOT GEMV arm
(`d0d188b3e`), the moe-glue LUT (`2c6ff6e22`) and the lm_head/router BFDOT
extension (`3a4262bc5`) on top of the same origin/main base. That branch head
is what "the FMA-contract draft" means in the tree and is what was measured.
The brief's `VT_CPU_PAGED_ATTN_BFDOT` attention K-dot arm exists on the same
branch and stayed OFF (the brief did not list it).

The third chain of the brief (the pure-torch transcription golden) has NO
logit-level row here: no per-position distribution for it exists in any
committed artifact, and running the torch transcription was out of scope for
this CPU-only pass. Its chain-identity distance to the primary, from the
committed goldens (`kolibri1_goldens.json` vs the primary fixture): prompts
2, 4, 7 token-identical for all 32 positions; prompt 1 diverges at position 1
(16/32 match); prompts 0 and 3 diverge at position 0 (0/32); prompt 5 at 5,
prompt 6 at 7.

## Results

Per-mode aggregate over the 202 aligned positions:

| mode | top-1 % | top-5 % | top-100 % | gap mean (nats) | gap med | cos mean | KL mean | KL med |
|---|---|---|---|---|---|---|---|---|
| prod  | 98.0 | 100.0 | 100.0 | 5.806 | 7.050 | 0.9814 | 7.070 | 8.755 |
| draft | 97.0 | 100.0 | 100.0 | 5.836 | 7.050 | 0.9812 | 6.971 | 8.214 |

Depth growth (mean per bin):

| mode | bin 0-5 (n=39) gap / KL / cos | bin 6-15 (n=62) | bin 16-31 (n=101) |
|---|---|---|---|
| prod  | 4.98 / 5.31 / 0.9775 | 5.14 / 6.57 / 0.9807 | 6.54 / 8.05 / 0.9833 |
| draft | 5.05 / 5.40 / 0.9772 | 5.16 / 6.55 / 0.9808 | 6.55 / 7.84 / 0.9830 |

Per-prompt mean gap/KL: prompts 0 and 7 sit at ≈0 nats on BOTH modes (the
echo prompts, where both sides are saturated on the emitted token); prompts
1-6 carry the whole distance (gap 6.3-10.1 nats). Per-position detail for
every position and both modes: `threeway_detail.json`.

## Findings

1. **Ranking: the two CPU modes are numerically indistinguishable from each
   other, and both are close to the primary in RANKING and far from it in
   CONFIDENCE.** The draft's KL is marginally lower (6.97 vs 7.07 mean,
   8.21 vs 8.76 median) and its gap marginally higher; the differences
   (≈0.1-0.5 nats) are an order below the depth trend and below the
   per-prompt spread. Neither mode dominates: at the logit level the
   FMA/NEON/BFDOT draft is NOT a regression against the scalar body, and the
   mode choice cannot be decided on this instrument — it is a tie within the
   instrument's own spread.

2. **The distance GROWS with depth on both modes**: mean gap 4.98 → 5.14 →
   6.54 nats, KL 5.3 → 8.1 from early to late positions. The growth is driven
   by the confident prompts (the echo prompts stay at ≈0 throughout), so the
   trend tracks where the distribution carries real entropy, not mere step
   count.

3. **The dominant distance term is a confidence (temperature-like) gap, not a
   ranking gap.** We place the primary's emitted token top-1 on 198/202
   positions (prod; 196/202 draft) and inside the primary's top-100 on
   100 % of positions, yet our logprob at that token is on average ≈5.8 nats
   HIGHER than the primary's own reported logprob, while cosine over the
   primary's top-100 support is 0.98. Our f32 logit rows are systematically
   SHARPER than the primary's reported distributions (a per-vector scale
   difference survives the log-softmax; an additive shift cannot). Explaining
   that sharpening — plugin logits processing, bf16 head rounding on the
   primary side, or an lm-head scale difference on ours — is the next
   traceable hypothesis and is NOT settled here.

## Caveats

- The KL is computed on the primary's truncated top-100 support with both
  sides renormalized; the primary publishes no tail, so it bounds nothing
  about the tail and underestimates true KL.
- The gap metric inherits the same truncation asymmetry only through the
  emitted token's own logprob, which the primary reports exactly; the gap is
  untruncated.
- One build host, one thread count (32), one run per mode; the scalar-arm /
  NEON-arm near-tie gates (W3, the decode bench) already bound the WITHIN-tree
  numerics spread, and the two modes' 0.1-nat separation here is consistent
  with that spread, not independently re-measured.
- Coverage: the primary scored 202/256 positions; the 54 dropped positions are
  exactly those where the primary's own emitted token missed its own top-100
  and are untestable at the logit level from this artifact.

## Reproduce

```sh
# capture (one build per tree; identical harness file)
VLLM_CPP_CPU_THREADS=32 VT_KOLIBRI1_DEQUANT_CACHE_MB=32768 \
  VT_THREEWAY_DUMP_DIR=/tmp/threeway/prod \
  /tmp/build-three-way/tests/test_kolibri1_threeway_capture
# draft tree adds: VT_CPU_PAGED_ATTN_NEON=1 VT_CPU_KOLIBRI_BFDOT_GEMV=1
python3 scripts/kolibri1-threeway-analysis.py /tmp/threeway/prod /tmp/threeway/draft
```
