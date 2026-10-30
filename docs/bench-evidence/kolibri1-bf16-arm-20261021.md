# Kolibri-1 bf16 arm: the plain bf16 checkpoint, and the control it measures (2026-10-21)

Row: MODEL-TEXT-kolibri-1 (`kolibri1-for-causal-lm`). Branch
`row/KOLIBRI1-BF16-ARM` (fresh from `origin/main` `845b3c59d`; spec
`.agents/specs/kolibri1-bf16-arm.md`, committed before the implementation).
Issue: ISSUE-LOCAL-01M5G38WAKAFKPNPQMSNM8EJ7H.

## What this change is for

The row's CPU forward runs the FP8 block-quantized checkpoint. Its divergence
from the primary oracle (the `aleph-alpha-inference` plugin's
`Kolibri1ForCausalLM`, the BFMMLA class) mixes two causes: the FP8 weight
quantization itself, and the rest of the numerics. The plain bf16 arm removes
the first cause by construction — a bf16 checkpoint's weights ARE the
dequantized values — so the same forward, on the same compute class, isolates
the quantization. What THIS note measured is the CONTROL leg of that: the
local copy on this box holds the dequantized values (the provenance verdict
below), so the two legs ran the same values and the measurement is a control
of the numerics. The upstream artifact's own gate, and the decomposition it
would produce, are OWED.

## THE PROVENANCE VERDICT (measured, not assumed)

The box carried TWO local copies of a bf16 Kolibri-1, and they are not the
same thing.

| copy | shard 1 size | shard 1 sha256 | what it is |
|---|---|---|---|
| `/mnt/models/Aleph-Alpha/Kolibri-1-bf16/` | 4,999,242,656 | (the campaign's own) | the output of `scripts/gen-kolibri1-bf16-reference.py`: the FP8 checkpoint DEQUANTIZED |
| `/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean/` | 4,998,628,528 | `8158b836a1ed57ac5b201fbb10c094a8660bc4d55bfd7c465eb9140b11988a36` | the same size as upstream, DIFFERENT bytes |
| `/mnt/models/Aleph-Alpha/Kolibri-1-BF16-upstream/` (fetched here) | 4,998,628,528 | `e626660074fd9ed3e3644f1d62f23be6adad9437e876f5f9b18260d8382d260b` | the upstream artifact: the sha256 IS the HF LFS object id (`x-linked-etag`) |

The upstream copy was fetched from
`https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16/resolve/cc6d729e6a024c1e59ad48887fdef95ff37b534d/model-00001-of-00032.safetensors`
and its sha256 matches the upstream LFS oid exactly.

The comparison, tensor by tensor (`/tmp/bf16-arm/compare-shard.py`, pure
Python over the safetensors container: the 8-byte header length, the JSON
header, and each tensor's `data_offsets` range):

```
python3 /tmp/bf16-arm/compare-shard.py \
  /mnt/models/Aleph-Alpha/Kolibri-1-BF16-upstream/model-00001-of-00032.safetensors \
  /mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean/model-00001-of-00032.safetensors
```

```
  tensors 1375 metadata {'format': 'pt'}      (both sides)
  names only in A: 0  only in B: 0
  header byte-identical: True
  metadata identical: True
  key order identical: True
  dtype differences: 0  shape differences: 0
  tensors with IDENTICAL bytes: 13
  tensors with DIFFERENT bytes: 1362
  sha256 A e626660074fd9ed3e3644f1d62f23be6adad9437e876f5f9b18260d8382d260b
  sha256 B 8158b836a1ed57ac5b201fbb10c094a8660bc4d55bfd7c465eb9140b11988a36
```

**The local "clean" copy is NOT tensor-identical to upstream.** The 13
identical tensors are exactly the ones that were never FP8 — the embedding,
the lm_head, the four norms per layer, the router gates, the router biases —
and the 1,362 that differ are exactly the FP8 linears. A second comparison
(`/tmp/bf16-arm/identify-local-copy.py`) settles what the local copy holds:

```
B and C share 1375 names; identical bytes: 1375, different: 0
```

`B` is the local "clean" copy and `C` is `/mnt/models/Aleph-Alpha/Kolibri-1-bf16`,
the campaign's own `gen-kolibri1-bf16-reference.py` output: **all 1,375 shared
tensors are byte-identical.** The local copy is the FP8 checkpoint's
DEQUANTIZATION, written into the upstream's file layout (same header, same
byte offsets, same total size — which is why its size matches upstream's and
its bytes do not).

Its index carries the FP8 checkpoint's `metadata.total_size`
(78,827,029,120) instead of the upstream's (156,206,149,120), and the upstream
index's tensor names and shard assignment are otherwise identical to the
local one's (58,353 tensors, 0 differences).

### What that means for the row's records

The primary oracle's per-position rows
(`tests/vllm/models/kolibri1_primary_topk_rows.json`) were generated on
`/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean` — the DEQUANTIZED model, not
the upstream bf16 model (`scripts/gen-kolibri1-primary-topk-rows.py`,
`MODEL = "/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean"`). The row's recorded
agreement numbers therefore compare our forward against the oracle's forward
ON THE SAME WEIGHT VALUES: they measure the numerics, and the FP8
quantization's own error is NOT in them.

## The measurement

One instrument for both arms:
`tests/vllm/models/test_kolibri1_threeway_capture.cpp` (teacher-forced on the
primary oracle's own chains, one full-vocabulary f32 logit row per scored
position) + `scripts/kolibri1-threeway-analysis.py` (the scorer), against the
primary oracle's published top-100 rows
(`tests/vllm/models/kolibri1_primary_topk_rows.json`). The capture takes the
checkpoint directory from `VT_KOLIBRI1_MODEL_DIR` (this change's addition to
the test) and the dump directory from `VT_THREEWAY_DUMP_DIR`; both legs ran
with `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384`, the exact-budget shape the
2026-10-19 adoption measurement used, so the dequant cache's derived budget
cannot move the compute class between the two legs.

```sh
# the fp8 arm (the released checkpoint)
env VT_KOLIBRI1_DEQUANT_CACHE_MB=16384 \
    VT_THREEWAY_DUMP_DIR=/tmp/bf16-arm/dump-fp8 \
    ./build/tests/test_kolibri1_threeway_capture
# the bf16 arm (the local bf16 copy)
env VT_KOLIBRI1_DEQUANT_CACHE_MB=16384 \
    VT_KOLIBRI1_MODEL_DIR=/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean \
    VT_THREEWAY_DUMP_DIR=/tmp/bf16-arm/dump-bf16-local \
    ./build/tests/test_kolibri1_threeway_capture
python3 scripts/kolibri1-threeway-analysis.py \
    /tmp/bf16-arm/dump-fp8 /tmp/bf16-arm/dump-bf16-local
```

### The numbers (256 scored positions: 8 prompts x 32 generated positions)

| leg | top-1 | top-5 | top-100 | gap mean (nats) | gap median | cos mean | KL mean (nats) |
|---|---|---|---|---|---|---|---|
| fp8 arm (`/mnt/models/Aleph-Alpha/Kolibri-1`) | 97.7% | 100.0% | 100.0% | 0.014 | -0.000 | 0.9983 | 0.044 |
| bf16 arm (`/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean`) | 97.7% | 100.0% | 100.0% | 0.014 | -0.000 | 0.9983 | 0.044 |

Per-prompt mean gap / KL, both legs: `p0 0.03/0.01, p1 -0.04/0.01, p2 -0.01/0.01,
p3 0.01/0.02, p4 -0.02/0.01, p5 -0.01/0.02, p6 -0.01/0.01, p7 0.16/0.26`.

**THE MEASURED DIFFERENCE: fp8 agreement minus bf16 agreement = 97.7% - 97.7% =
0.0 percentage points, and it is a CONTROL of the numerics, NOT the fp8
weight quantization's decomposition (the next section says why). The worst gap
is 0.014 nats in both legs; the hard flips (top-1 misses) are 6 of 256 in
both legs.**

The two legs are not merely equal in the aggregate: their 256 per-position
logit rows are BYTE-IDENTICAL, prompt by prompt.

```
prompt0 IDENTICAL c112b6f68dcb33ad179965d87b234469e21ca6a6c44237720c8dd553326feeaf
prompt1 IDENTICAL 03d3175a258f60e7b022358f03f31127b3b46496dda06580f5524b9248272caa
prompt2 IDENTICAL 774351baf854ad5143b3ca9f2dac112baa0f3f536902f5d1a03923b7369d2da0
prompt3 IDENTICAL 11027d1ff55cb2c21be1e8716a2902f41b7279a66ed65f6b4adc2c3c0d443198
prompt4 IDENTICAL 1ed845590659d89cbbb008fdb04a829920c0ddd325520679fc43c33db56be793
prompt5 IDENTICAL 78bde090ff13c6157149f3853662ece1b41ec3f47d9e094e061832ebbeabd12b
prompt6 IDENTICAL 50475e6d3ce11f3cff61121ae7973d43cdfbab8dce49b1c5228c5091ba80aa12
prompt7 IDENTICAL 27b5e7deff26284cbe0a30dbb08519948786cb2291f913d9b0b85be88d737e67
```

### What the measured ZERO means

It is a CONTROL, and the provenance verdict says why. The local bf16 copy
holds the DEQUANTIZED values, byte for byte (measured above), so the two legs
run the SAME weight values through the SAME compute class; the measurement
therefore shows what the change set out to show about the NUMERICS — the
plain bf16 arm and the fp8 arm are the same forward — and it shows NOTHING
about the quantization's error, because the denominator (the primary oracle's
rows) was computed on the same dequantized weights.

Two things follow, and both are the point of this change:

1. **The bf16 arm is reachable and correct.** A plain bf16 Kolibri-1
   checkpoint loads through the real loader, the real CPU forward runs it, and
   it produces the primary oracle's own distributions to the last bit of the
   f32 logit row. The unit gate says the same on the tiny fixture: the fp8
   fixture and its exact bf16 dequantization give a worst logit gap of 0 at
   the prefill shape AND at the decode shape.
2. **The FP8 quantization's own contribution is still not separated**, and
   the reason is now a measured fact rather than an assumption: the primary
   oracle's rows were generated on the dequantized model, not on the upstream
   bf16 model. The decomposition needs a reference computed on the UPSTREAM
   bytes, and the upstream artifact's 152 GB did not land in this session
   (see below). It is OWED, and it is the next measurement this row owes.

The bf16 arm is NOT worse than the fp8 arm (they are the same measurement),
so the row's red line ("adoption may not move AWAY from the primary") holds.

## What was NOT measured here

- **The upstream artifact's own gate.** The upstream copy is STILL FETCHING on
  this box, and the number moves: re-measured at 2026-10-22 12:21 CEST,
  shards 1-10 are complete (shard 1 verified against the upstream LFS oid),
  shard 11 is in flight, and 61.73 GB of the 152 GB are on disk. An earlier
  revision of this note said "the fetch has NOT moved since 2026-10-22 01:35
  (the same 43 GB)"; that was WRONG and is corrected here — the fetch moved,
  and it is still running. It runs at ~12 MB/s with 48 parallel range
  requests, so the full copy needs ~3.5 h and did not land in this session.
  The upstream copy's OWN run against the primary rows (the real
  decomposition) is therefore still OWED, and the number above is a LOWER
  BOUND on what is on disk when this note is read.
- **The decomposition against an upstream-computed reference.** It needs the
  primary oracle to re-emit its rows on the upstream bytes (the oracle's
  environment is not on this box), and then the upstream copy's run here.
  Both are follow-ups, recorded in the spec's `## Owed`.
- **The n-gram spec-decode e2e gate** (`test_kolibri1_ngram_spec`) is RED in
  this session, and this change does not cause it: the gate demands that the
  spec-ON chain equal the spec-OFF chain byte-for-byte, and the spec-ON leg's
  verify rows run at t > 1 (the f32-widen tier) while the spec-OFF leg's steps
  run at t == 1 (the bf16-dot tier) — the same class boundary the 2026-10-19
  bf16-dot adoption recorded as the spec probe's tier floor (6/8, see
  `docs/bench-evidence/kolibri1-bf16dot-adoption-20261019/report.md`). The
  failing gate's own message reproduces the probe's recorded baseline: "W3
  relationship: 4/8 prompts token-identical to the torch golden end to end
  (probe: 4/8)". This change leaves the fp8 arm's class selection untouched
  in the configuration that ran, so the gate's outcome is the base's outcome.

## Reproduce

```sh
cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build -j 8
ctest -R kolibri1 --output-on-failure      # the focused family
./build/tests/test_kolibri1_bf16_arm         # this change's own gate
```

The two capture legs and the scorer are the commands above; the scorer reads
`tests/vllm/models/kolibri1_primary_topk_rows.json` from the repository root
and writes `threeway_detail.json` there.

## Environment

Host: aarch64 (Ampere-1a class, FEAT_BF16 through the HWCAP2 probe), 128
cores, 255 GiB RAM, a CONTENDED shared box (other sessions ran device
measurements throughout). The build and the gates ran pinned to CPUs 0-15
with `nice -n 19`; the two capture legs ran on CPUs 0-15 and 16-31
respectively. The upstream fetch used 48 parallel HTTP range requests.
