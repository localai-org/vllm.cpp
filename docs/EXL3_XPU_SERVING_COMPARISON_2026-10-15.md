# Bounded C++ / Python EXL3 serving comparison, 2026-10-15

Measured native product: `ac3d57bf4`; measured client source:
`677856acdc701ec7be1606420a104d3ad1931abf`. These original evidence IDs remain
provenance after the user-requested history cleanup. The native product is
reconstructed as `4c5d2386bf42925ae64b482889ac283cd91d778a` directly on Main
`50d425bec`; the client is `a43b5bb1a9d4554902b96e3934161a6babdef845`.
Each reconstructed commit has the exact same tree as its source snapshot.

The first Main reconciliation used the actual prior content base to reduce
115 misleading graph conflicts to 3 real conflicts. Its initial forward push
retained duplicate upstream ancestry, making the PR commit list misleading.
The user then requested rewriting the feature branch onto canonical Main.
The cleaned history contains only four feature/observer/report commits and
is published with an explicit expected-head force-with-lease. The original
history is backed up. Before this documentation correction, the complete
reconstructed tree was exactly `ad0bfe95edc819bfe78dcb4a4ca8bc4aad609d95`,
identical to the prior report head `8e11beb54`. No product or test content
changes in the history cleanup. Model-aware ngram dispatch and EXL3 MTP
page sharing are both retained.

## Executed result

C1, exactly 4096 identical prompt ids and 256 output tokens, greedy/ignore_eos,
FP16 activations, FP8 KV, max batched tokens 4096, prefix cache off and identical
65,536-token draft mapping. Each arm has a separate 32-token primer and four
timed requests. These are medians of the last three, not startup/cold samples.
The ordinary completion request includes `logprobs: 0` in both engines.

| Mode | C++ prefill s | Python prefill s | C++ decode tokens/s | Python decode tokens/s | C++ total s | Python total s |
|---|---:|---:|---:|---:|---:|---:|
| Target only | 1.984 | 1.462 | 25.72 | 29.91 | 11.898 | 9.992 |
| MTP3 | 2.055 | 1.472 | 58.76 | 68.67 | 6.394 | 5.190 |

Native decode throughput is -14.01% relative to Python without MTP
and -14.42% with MTP3. Native prefill duration is
+35.70% / +39.60%; total duration is
+19.08% / +23.20% respectively. Decode rate is 255 tokens divided by
the measured post-first-token request phase. These are serving-phase timings,
not isolated kernel times or a measurement of interpreter overhead.

MTP counters over all four timed requests (primer excluded): native accepted
708 / 972 proposals (72.84%), Python 704 / 984
(71.54%). Counters can include terminal overshoot. MTP improves decode by
2.285x native / 2.296x Python. Lower native acceptance does not
explain this case. Both the roughly 0.58s prefill and 0.63s decode excess contribute
to the roughly 1.20s MTP3 total gap; a causal profile is still needed.

All 16 decoded outputs are identical, within and across modes/engines. Raw output
ids, logits and states were not compared; public token representations differ
between implementations. All 11 checkpoint files, including both weight shards,
were freshly hashed against the installed release. Prompt ids and draft-map
bytes match. No preemptions occurred.

## Runtime and qualification scope

Python uses the exact installed production image `36e41ea02775` with vLLM 0.31.0,
Torch 2.14.0+xpu and oneDNN 3.13.4. Native uses the retained oneAPI 2026.1.1 builder
and oneDNN 3.13.0. Libraries, drivers, power and weights were not changed. This is
a current-stack comparison, not a rerun of the older arithmetic oracle.

Both workers were isolated/sequential on B70, limited to 2 CPUs / 24 GiB without
swap, C1 admission and 262144 configured context. Native requests 1600-token KV
pages / 180 blocks; Python exposes effective 1600-token pages with 228/201 blocks.
Python changes its Mamba mode to `none` with prefix caching off; native keeps
compact align ownership. These implementation/capacity differences are retained,
not hidden behind a claim of identical internal state geometry.

Use the [native build/runtime guide](EXL3_XPU_VISION_SERVING.md) with C1,
`--max-num-batched-tokens 4096`, prefix caching disabled and served name
`Qwen3.8-27B`. Use the same overrides on the pinned Python profile, retaining
its guarded attention and M04 implementation. Python uses FULL_DECODE_ONLY
capture sizes1/2/4. Both allow two images and no video, although this workload
contains no media. On each separately launched server, run the shared client:

```sh
python3 -O scripts/mm/measure_native_exl3_text_regression.py \
  --url http://127.0.0.1:PORT --model Qwen3.8-27B \
  --spec-depth 0 --output NEW_TARGET_RESULT.json
python3 -O scripts/mm/measure_native_exl3_text_regression.py \
  --url http://127.0.0.1:PORT --model Qwen3.8-27B \
  --spec-depth 3 --output NEW_MTP3_RESULT.json
```

Use different local ports or let the old listener's TIME_WAIT state expire
before the next Python launch. Keep the source images and recipes fixed;
the client checks within-arm repetition, not cross-engine numerical equality.

The merged native build passes. Focused checks pass: KV seam 43 assertions,
FP8 budget 6, vision operators 35 cases/1,854 assertions, synthetic graph/eager
state/owner test 4,737 assertions. Seven public MTP3 PNG/JPEG, SSE and ordered
two-image requests pass with 300 actual proposed drafts. Three usage-admission
unit tests preserve strict integer counts, missing/wrong-count rejection and
optional current API metadata. No full preflight/CI campaign was run.

Earlier failures remain in the external evidence: the aborted graph merge;
the incomplete synthetic FP32 fixture; Python M0 startup port reuse; and the
older usage observer rejecting the newer nullable-detail schema. The v2 failed
response was not retained; v3 captures the actual counts/details. No earlier
failed producer was relabelled successful. The observer now retains rejected
responses and the already measured before/after counters.

The companion JSON contains binary/source/config ids, individual samples,
raw-evidence hashes and the explicit frozen contract. Large tensors, weights
and local launch paths are not copied into this public projection.

## Next bounded work

A fresh optimization plan should use this source and these measurements. First
profile this same workload with timing instrumentation kept separate from scores,
then compare prefill dispatch and per-target/draft/verify costs. Python logs show
guarded oneDNN prefill on 4096 rows; that is a route observation, not yet a cause
of the whole gap. Preserve numerical, ownership and actual MTP-state invariants.
Do not reopen completed vision integration or rerun a broad benchmark matrix.

The one-prompt C1 result does not qualify long contexts, concurrency, image
throughput, stochastic sampling or complete Python vision features. The existing
PR remains Draft. Production stays stopped and temporary workers are removed.
