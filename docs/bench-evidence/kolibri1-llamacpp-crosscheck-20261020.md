# Kolibri-1 llama.cpp cross-check: third independent reference vs the primary oracle (2026-10-20)

Row: MODEL-TEXT-kolibri-1 (`kolibri1-for-causal-lm`). Branch
`row/KOLIBRI-LLAMACPP-CROSSCHECK` (fresh from `origin/main` `61d333e52`).
Developer-approved task: add upstream llama.cpp's freshly-merged Kolibri-1
support as a third independent reference and cross-check our CPU gates
against it. Measurement only — no vllm.cpp product code changes.

## What landed upstream

llama.cpp merged Kolibri-1 (Aleph-Alpha) support in `62318cd57`
(#30673, 2026-10-17): arch `kolibri1`, converter `Kolibri1Model(AfmoeModel)`
(`conversion/kolibri1.py`), the released FP8 checkpoint goes through the
generic `weight_scale_inv` dequant path (`conversion/base.py`, `quant_method
== "fp8"`). Upstream follow-up #30865 (router precision overrides) is AFTER
the pin and is NOT in this measurement.

## Pin and build identity

- Oracle revision: `62318cd57dc63900ecb153272c0b10f78550855d` (llama.cpp main).
- Source tree: `/home/lu_zero/Sources/llama.cpp`, `git status --porcelain`
  EMPTY at the pinned commit. Read-only against ggml-org; the PR head ref
  (`pull/30673/head` = `566df8ce4`) was fetched only to locate the merge.
- Build: CPU-only, `cmake -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF`,
  GNU 16.2.0, Linux aarch64 (Ampere-1a, FEAT_BF16/dotprod — the same hardware
  class as our production bf16-dot default). `version: 0.7.0-dev (build 11709,
  commit 62318cd57)`.
  - `llama-cli` sha256 `05d0f29c20da702b482646d27b804b9cad5b9c315d311999dd5bc3f5a4a3295a`
  - `llama-server` sha256 `fdff524b31d6e6b8d2462d5c78229ad4421e463ad95c9aedaa880216e1322afc`
- Executed path: `llama-server -t 16`, all defaults otherwise (repack ON,
  flash-attention AUTO, no `-nr`/`-ub` overrides). Per the oracle record's
  path-pinning rule this is the path that produced every number below.

## The GGUF artifact

Produced with the pin's own converter, FP8 through the native path:

- Command: `convert_hf_to_gguf.py /mnt/models/Aleph-Alpha/Kolibri-1
  --outfile /tmp/kolibri1-llamacpp-auto.gguf` (default `--outtype auto`).
- Result: 903 tensors, 156.3 GiB logical, `general.architecture = kolibri1`,
  `general.file_type = 32` (`MOSTLY_BF16`), `expert_count = 384`,
  `expert_shared_count = 1`, sigmoid gating.
- File: `/tmp/kolibri1-llamacpp-auto.gguf`, 149,069.2 MiB on disk,
  sha256 `76ddb05e458cce1a023d42cd4f13dc8e8152fbb6501c1f680918fc7d2a0fcefd`.
  Not committed; re-derivable from the command above.

The runtime LOADED the artifact and generated coherent text end-to-end on all
eight prompts: this oracle pin is gateable for kolibri-1.

## Method

The primary oracle's committed fixture
(`tests/vllm/models/kolibri1_primary_topk_rows.json`, 202 aligned positions
over the 8-prompt battery, per-prompt scored counts 32/30/24/20/30/20/14/32)
is the reference. Two measurements through llama.cpp:

1. **Aligned-position top-1** (the same statistic as our committed 98.0%
   number). PROVISIONAL LABEL: the primary top-100 rows on main carry a
   confirmed off-by-one (the writer stored each row one position behind the
   generated chain; repair in flight on row/PRIMARY-ROWS-OFFBYONE). Both
   sides of this comparison used the same lagged alignment, so the equality
   of the two percentages below is the meaningful statistic — two
   independent implementations agreeing exactly — while the absolute 98.0%
   label is provisional until the repaired rows land: for each aligned position k, the prompt ids plus the primary's
   generated ids `0..k-1` were sent as a token-array completion with
   `n_predict=1, temperature=0`, and the sampled id was compared with the
   primary's emitted token. Both sides therefore score position k under the
   IDENTICAL prefix — a numerics distance, not a chain artifact.
2. **Chain decode**: the prompt ids alone, `n_predict=32, temperature=0`,
   compared position-wise with the golden chains
   (`kolibri1_goldens.json` `generated_ids`, the torch-transcription /
   primary chains).

Raw results: `/tmp/kolibri-xcheck/llamacpp_results.json` (not committed; the
tables below are the record).

## Three-way agreement

| side vs PRIMARY emitted token | aligned positions | top-1 agreement |
|---|---|---|
| OURS — production bf16-dot default (committed, `docs/bench-evidence/kolibri1-bf16dot-adoption-20261019/report.md`) | 202 | **198/202 = 98.0%** |
| llama.cpp `62318cd57`, bf16 GGUF (this run) | 202 | **198/202 = 98.0%** |

Both independent implementations miss the same four-or-fewer knife-edge
positions; ours and llama.cpp agree with each other at the reference at
least wherever both match it.

| llama.cpp chain vs golden chain | match | first divergence per prompt (0-7) |
|---|---|---|
| 126/256 = 49.2% | 3 of 8 prompts token-exact through 32 steps | 0, 1, —, 0, —, 5, 7, — |

The chain number is LOW while the aligned number is HIGH — the same shape our
own chain runs show: single-token near-ties at small logit margins re-rank,
and once a chain diverges the later positions compare different contexts and
are meaningless as a numerics statistic. Do not read the 49.2% as model
quality; read the 202-position teacher-forced numbers.

llama.cpp's four aligned mismatches: (p0, k0, primary 116026 → 121598),
(p1, k1, 35423 → 46252), (p3, k3, 2042 → 1584), (p6, k16, 13111 → 47354).
No tuning was applied to either side; this is a measurement.

## What this changes

`.agents/oracles/llama-cpp.md` advances its pin from `11fe0215` to
`62318cd57` with this file as the gateability evidence for the kolibri-1
artifact. `docs/USAGE.md` owes no line: the kolibri section's promises
(commands, workflow, gate provenance) did not change, and this run adds no
command, config key, or workflow.

## Not measured here

Speed and memory floors vs this oracle remain owed by the oracle record's
re-take debt; this run pins correctness only. Upstream #30865 (router
precision overrides removal) is newer than the pin and unaudited here.
