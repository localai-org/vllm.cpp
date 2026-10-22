# Spec: the Kolibri-1 bf16 arm — the fp8 weight quantization's own error, measured

Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
Status: ACTIVE (spec committed before implementation; one PR carries the spec and
the implementation — the repository default, `.agents/developer-preferences.md`
§Git integration).
Issue: ISSUE-LOCAL-01M5G38WAKAFKPNPQMSNM8EJ7H (local, open until this change
lands).
Siblings: `.agents/specs/kolibri-1-cpu.md` (the row),
`.agents/specs/kolibri1-bf16-dot-cpu.md` (the compute class this arm rides),
`.agents/specs/kolibri1-decode-bench-near-tie-instrument.md` (an arm of this
row, specced in the same shape).

## Problem

Our CPU forward runs the fp8 block-quantized Kolibri-1 checkpoint, and it
diverges from the primary oracle (the `aleph-alpha-inference` plugin's
`Kolibri1ForCausalLM`, BFMMLA class) on a measurable fraction of the
teacher-forced positions. That divergence has two candidate causes and the
tree cannot separate them:

1. the fp8 weight quantization itself (every linear weight is an F8_E4M3
   block with an f32 `weight_scale_inv` sibling), and
2. the rest of the numerics (the accumulation order, the activation dtype, the
   attention dot, the router).

The bf16 arm removes cause 1 by construction: an upstream bf16 Kolibri-1
checkpoint's weights ARE the dequantized values, so the same forward on the
same compute class isolates the quantization's own contribution.

Today the arm does not exist. `kolibri1_weights.cpp::ResolveKolibri1WeightScheme`
(the scheme dispatch that replaced the old `CheckQuantConfig`)
requires the checkpoint's `quantization_config` to be fp8 block [128, 128]
dynamic and refuses everything else by name, with the message "a bf16
Kolibri-1 artifact is not an implemented arm". The upstream bf16 artifact
(`Aleph-Alpha/Kolibri-1-BF16`, revision
`cc6d729e6a024c1e59ad48887fdef95ff37b534d`) carries no
`quantization_config` at all, so the loader rejects it and the decomposition
cannot be measured.

## Scope

In: the CPU loader's scheme dispatch (fp8 block [128, 128] dynamic — the
existing arm — and plain bf16 — the new arm), the CPU forward's bf16 path,
the tests, the measured evidence, and the artifact record.

Out: the GGUF arm, the CUDA and Tenstorrent arms, the tokenizer, the router
and attention math (unchanged), and any change to the fp8 arm's bytes.

## Upstream anchors

- The bf16 artifact: `Aleph-Alpha/Kolibri-1-BF16` @
  `cc6d729e6a024c1e59ad48887fdef95ff37b534d`, 32 safetensors shards,
  `model.safetensors.index.json` `metadata.total_size = 156,206,149,120`.
  Its `config.json` carries no `quantization_config` and no `torch_dtype`; it
  carries `dtype: bfloat16` and `head_dtype: float32`, exactly like the fp8
  checkpoint's config minus the quantization block.
- The tensor layout is the fp8 checkpoint's layout minus the
  `weight_scale_inv` siblings: 58,353 tensors against the fp8 checkpoint's
  116,303 (= 58,353 + 57,950 scales). The names and the shard assignment are
  identical to the upstream index's (verified against the upstream index
  itself, not inferred).
- The primary oracle: `.agents/oracles/aleph-alpha-inference.md` (the
  `aleph-alpha-inference` plugin registers `Kolibri1ForCausalLM`). vLLM is
  the primary oracle wherever it implements the behavior; it does not
  implement `kolibri1` at the pin, so this architecture's oracle is the
  model author's own plugin, and the gate rows it published
  (`tests/vllm/models/kolibri1_primary_topk_rows.json`) are the denominator.
- The compute class this arm rides: SPEC `kolibri1-bf16-dot-cpu` and
  `src/vllm/model_executor/models/kolibri1_bfdot_gemv.cpp` — the bf16-dot
  GEMV (`vbfdotq_f32`, bf16 in / f32 accumulate), feature-gated on
  `DetectArmBf16()`.

## Design

### 1. The loader: a scheme dispatch, not one hard requirement

`CheckQuantConfig` becomes `ResolveKolibri1WeightScheme`, which returns the
checkpoint's scheme and refuses every other scheme BY NAME:

| scheme | how it is recognized | what the loader does |
|---|---|---|
| `kFp8Block` | `quantization_config` with `quant_method: fp8`, `activation_scheme: dynamic`, `weight_block_size: [128, 128]` | unchanged: the raw F8_E4M3 bytes plus the f32-widened scale grid |
| `kBf16` | NO `quantization_config` at all | the BF16 weights are read as they are; there is no scale sibling and no dequant |
| anything else | anything else | refused by name, and the message names the missing arm |

The fp8 arm's behaviour and its messages are unchanged: the same three
`quantization_config` checks, in the same order, with the same text.

The accounting is scheme-aware. `EnumerateKolibri1Tensors` and
`AccountKolibri1Tensors` take the scheme, because the bf16 arm's expected
tensor set has no `weight_scale_inv` siblings. A bf16 checkpoint that ships a
scale sibling is therefore UNACCOUNTED and refused by the accounting pass,
and an fp8 checkpoint that is missing one is MISSING and refused by the same
pass. The two arms cannot be confused with each other.

The per-tensor dtype check is scheme-aware too: under the bf16 scheme a
linear weight that ships `F8_E4M3` is refused by name (a bf16 checkpoint
with an fp8 weight is contradictory, and loading it would silently pick the
fp8 arm for that projection).

### 2. The forward: the same compute class, no second numerics path

The bf16 arm's weights are already bf16, so:

- NO dequant, NO scale application, NO dequant cache. The dequant cache
  exists to hold the fp8 arm's DECODED bytes; the bf16 arm has nothing to
  decode, so the cache does not engage for it at all. The loader's bf16
  projections are read with the same `LoadBf16Direct` the fp8 arm's router
  gate already uses, and the forward resolves them as RESIDENT bf16.
- The SAME compute class as the fp8 arm's dequantized weights: the bf16-dot
  GEMV (`kolibri1_bfdot_gemv::Gemv`, bf16 in / f32 accumulate) at the
  decode shape, and the shared `vt::MatmulBT` / `vt::MatmulBTBatch` f32
  widen at every other site. There is no second bf16 kernel.

The class selection is ONE rule, and BOTH arms read it:

```
bf16-dot class  <=>  the weight is bf16-resident
                 &&  the decode shape is t == 1
                 &&  the output is bf16
                 &&  DetectArmBf16() proved FEAT_BF16
                 &&  the dequant cache is enabled
```

The last conjunct is the fp8 arm's recorded residency precondition: the
bf16-dot GEMV is the arm that multiplies the WARM bf16 bytes, and the
dequant cache is what makes the fp8 arm's bytes warm (its budget is derived
from the host's memory report, so the row's numerics are ALREADY
host-dependent — `docs/ENVIRONMENT.md`). The plain bf16 arm has NO cache of
its own: there is nothing to decode, so the cache never holds or serves one
of its bytes — the resolve returns the resident tensor and no lease. The
conjunct is the ROW's rule, read by both arms, so the two arms can never
diverge on the same site: the same bytes always ride the same class, in
every configuration.

The fp8 arm's class selection is otherwise UNCHANGED: the same three
conjuncts it had before this change, in the same order, with the same
meaning. The change is that the bf16 arm's weights now reach the SAME
selection instead of a separate `LinearBTRaw` path.

An earlier draft of this spec REMOVED the cache conjunct, on the argument
that the class is a property of the bytes and the hardware and must not move
with a caching decision. The gate disagreed, and the gate is right: the
n-gram spec-decode e2e gate compares a spec-ON chain (whose verify rows
run at t > 1, the f32-widen tier) with a spec-OFF chain (whose steps run at
t == 1, the bf16-dot tier) and demands them byte-for-byte equal. That
demand holds only while the t == 1 class does not move under the cache
knob. The conjunct stays, and the two arms read the same rule.

### 3. The artifacts

Two local copies of a bf16 Kolibri-1 exist on this box, and they are not the
same thing:

- `/mnt/models/Aleph-Alpha/Kolibri-1-bf16/` — the campaign's OWN fp8→bf16
  dequantization (`scripts/gen-kolibri1-bf16-reference.py`). Its shard 1 is
  4,999,242,656 bytes, NOT the upstream size. It is not an upstream copy and
  it is not used by this change.
- `/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean/` — shard 1 is 4,998,628,528
  bytes, exactly the upstream size, but its sha256 is
  `8158b836a1ed57ac5b201fbb10c094a8660bc4d55bfd7c465eb9140b11988a36` against
  the upstream LFS oid
  `e626660074fd9ed3e3644f1d62f23be6adad9437e876f5f9b18260d8382d260b`. Same
  size, DIFFERENT bytes. Its index carries the FP8 checkpoint's
  `metadata.total_size` (78,827,029,120) instead of the upstream's
  (156,206,149,120), so the index itself is a rewrite.

A safetensors header is JSON whose key ORDER changes the bytes without
changing the length, so a re-save of the same tensors in a different order
would produce exactly this size/digest mismatch — but a different revision
would produce it too. The verdict is MEASURED, not assumed: the upstream
shard 1 is fetched and compared with the local copy TENSOR BY TENSOR (every
tensor's bytes, its dtype and its shape) plus the header. The verdict and
its numbers are in the evidence note.

## Tests

Red first, in this order; the red capture is in the commit body.

1. **The bf16 arm loads and the forward produces logits.** A synthetic bf16
   checkpoint (the W1 fixture's geometry, every linear weight BF16, no scale
   siblings, no `quantization_config`) loads through the real loader and the
   real CPU forward produces a logits row. RED before this change: the
   loader refuses it by name.
2. **The fp8 arm is unchanged.** The same synthetic fp8 checkpoint still
   loads, with the fp8 block arm and the f32-widened scale grid, exactly as
   before.
3. **An unknown scheme is refused by name.** A checkpoint whose
   `quantization_config` names a scheme that is neither fp8-block nor bf16
   is refused, and the message names the missing arm.
4. **The two arms agree on the same bytes.** A synthetic fp8 checkpoint and
   its EXACT bf16 dequantization (the same values, the same shapes) produce
   the same logits, because the same bytes must ride the same compute class.
   The bf16 arm must not be WORSE than the fp8 arm; a worse result is a
   failure, not a tolerance to widen.

## Gates

- The Kolibri-1 CPU test family (`ctest -R kolibri1`) is green.
- The record checkers (`check-agent-record.py`, `check-env-doc.py`,
  `check-site.py`, `check-commit-style.py`, `check-commit-trailers.py`) pass.
- The MEASUREMENT (the number this change exists to produce): the primary
  oracle's per-position top-100 rows
  (`tests/vllm/models/kolibri1_primary_topk_rows.json`) are the denominator,
  and BOTH arms are scored against them with the same instrument
  (`tests/vllm/models/test_kolibri1_threeway_capture.cpp` +
  `scripts/kolibri1-threeway-analysis.py`, the shape the fp8 arm's standing
  cross-check uses, `docs/bench-evidence/kolibri1-llamacpp-crosscheck-20261020.md`).
  Reported side by side: the agreement count, the worst gap, the hard flips,
  and the DECOMPOSITION — the fp8 arm's agreement minus the bf16 arm's
  agreement, which is the fp8 weight quantization's own contribution. If the
  bf16 arm is not at least as good as the fp8 arm, that is a FINDING to
  report, not to hide.

## Evidence

`docs/bench-evidence/kolibri1-bf16-arm-20261021.md`: the provenance verdict
with its command and numbers, the two arms' agreement counts, the worst gap
and the hard flips for each, and the decomposition.

Measured on 2026-10-21 (256 scored positions, the primary oracle's rows as
the denominator): the fp8 arm and the plain bf16 arm both score 97.7% top-1
(6 hard flips of 256), a 0.014-nat mean gap, 0.9983 mean cosine — and their
per-position logit rows are BYTE-IDENTICAL. The decomposition is therefore
0.0 percentage points, and the note says why: the local bf16 copy holds the
DEQUANTIZED values (measured), so the two legs are a CONTROL of the numerics,
not the quantization's decomposition. The upstream artifact's own gate is
OWED.

## Risks

- **A second numerics path.** The bf16 arm must ride the fp8 arm's compute
  class. Mitigated by test 4: the same bytes must produce the same logits.
- **The scheme dispatch accepting a wrong checkpoint.** Mitigated by the
  scheme-aware accounting (the bf16 arm's expected set has no scale
  siblings, so a mixed checkpoint is refused) and by the per-tensor dtype
  check.
- **The fp8 arm's numerics moving.** They do not: the class selection's
  three pre-existing conjuncts are unchanged, and the only new term is the
  `out_dtype == kBF16` one, which the fp8 arm already satisfied at every
  site the bf16 arm reaches. The decode bench's byte-exact anchor (11622 on
  the bf16-dot tier) is the proof that the fp8 arm's decode stream did not
  move.
- **The provenance of the local bf16 copy.** Settled by measurement (the
  tensor-by-tensor comparison), not by the byte count. If the local copy
  differs from upstream, the arm is gated against the upstream copy and the
  local one is refused by name.

## Stop conditions

- If the bf16 arm's agreement with the primary oracle is WORSE than the fp8
  arm's, STOP: report the finding, do not widen a band, do not re-tune.
- If the same-bytes test cannot be made to pass, the two arms do not share a
  compute class: that is a defect in this change, not a tolerance to widen.
- If the upstream shard cannot be fetched, the functional work proceeds on the
  local copy and the PROVENANCE verdict stays OWED; the provenance claim is
  then reported as NEEDS_CONTEXT and nothing is claimed about it.

## Owed

- The GGUF arm (the row's standing debt) and the CUDA/Tenstorrent bf16 arms:
  unchanged by this change and still owed by the row.
- The primary oracle's own bf16 rows were generated on the LOCAL bf16 copy
  (`scripts/gen-kolibri1-primary-topk-rows.py`, `MODEL =
  "/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean"`). If the provenance verdict
  is "the local copy differs from upstream", the primary rows are tied to
  that copy and the denominator itself needs a re-emission against the
  upstream bytes. That is a follow-up, and it is recorded here rather than
  silently absorbed.
- The bf16 arm's numerics follow the row's existing host-dependent
  warm-bytes condition (the dequant cache's derived budget), because both
  arms read the SAME class rule. A host that cannot afford the cache's knee
  runs the f32-widen tier for both arms. This is the row's existing
  contract, not a new one.
- The dequant cache's budget policy was derived for the fp8 arm's 74 GB
  checkpoint. The bf16 arm's 152 GB weights have no cache to size, and the
  host's memory report under a 152 GB page-cache working set is a separate
  question this change does not answer.
