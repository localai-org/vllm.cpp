# Flint GGUF arm — the published containers load and their chains replay

Row: `MODEL-TEXT-FLINT` (wave 2: the GGUF arm + the real-weights token gate;
W1 spec [.agents/specs/flint-cpu-gguf.md](flint-cpu-gguf.md) names both under
its `## Owed`)
Issue: ISSUE-LOCAL-01M5PRSC2E4VR54HBSWFPMN37E
Branch: `row/MODEL-TEXT-FLINT-GGUF`, base `7f842ee9b3c7e9b651283de48f42cd76a0ad5d30` (origin/main)

## Scope

The `flint` GGUF arm on the CPU path, mirroring the kolibri1 GGUF precedent
(commits 71ee347e9 + 0350841b7, spec
[.agents/specs/kolibri1-gguf-arch.md](kolibri1-gguf-arch.md)):

- the arch row in the top-level GGUF dispatch table (`kGgufArchArms`,
  `src/vllm/entrypoints/model_loader.cpp`);
- the weights TU that loads the published containers into the EXISTING
  `FlintWeights` the safetensors arm fills
  (`include/vllm/model_executor/models/flint.h`), so the production forward is
  unchanged;
- the registry load path accepting `ModelSource::Kind::kGguf` (W1 refused it
  by name);
- the `flint` pre-tokenizer mapping in `Tokenizer::FromGguf`
  (`src/vllm/tokenizer/tokenizer.cpp`) plus its own staged split pattern
  (`SplitPattern::kFlint`) — the PR #30928 regex set ported verbatim as three
  stages, NOT aliased onto an existing pattern;
- the REAL-WEIGHTS token gate: golden greedy chains captured from the pinned
  oracle on the published Q4_K_M container, replayed through the production
  registry entry (`ModelRegistry::Resolve/Load/Forward`).

Out of scope: the paged-engine arm (W1's `## Owed` keeps it), the Tenstorrent
arm, keep-quant residency, and any device arm.

## Upstream anchors

- **llama.cpp PR #30928 head `1b44e30775dcb2f93952116b05f89e5bd797fb00`**
  ([.agents/oracles/llama-cpp-flint.md](../oracles/llama-cpp-flint.md)), the
  only llama.cpp that converts, loads and runs flint. Tensor names
  (`blk.N.attn_{q,k,v,output}`, `attn_{q,k}_norm`, `attn_norm`, `ffn_norm`,
  `attn_sinks`, `ffn_gate_inp`, `ffn_{gate,up,down}_exps`,
  `ffn_{gate,up,down}_shexp`, dense `ffn_{gate,up,down}`), the KV names
  (`flint.attention.{sliding_window,rope}_pattern`,
  `flint.leading_dense_block_count`, `flint.expert_*`), and the `flint`
  pre-type regexes (`LLAMA_VOCAB_PRE_TYPE_FLINT`, `src/llama-vocab.cpp:284`)
  are the reference. UNLIKE the kolibri1 container, the flint containers ARE
  llama.cpp-conformant (`expert_gating_func = 2` = `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`,
  `src/llama-hparams.h:19`), so llama.cpp is a runnable floor here and the
  goldens are its own greedy output.
- **vLLM implements nothing for flint** (W1 recorded the zero code hits).

## The published containers (read 2026-10-24, on-box, gguf headers)

| Container | ggml types | verdict |
|---|---|---|
| `flint-GGUF/flint-Q4_K_M.gguf` (23 GB) | Q4_K, Q6_K, F32 | ADMITTED (the gate vehicle) |
| `unsloth-flint-GGUF/flint-preview-UD-Q4_K_XL.gguf` (24 GB, sha256 `63c2ecc4...` per W1 spec) | Q4_K, Q5_K, Q8_0, IQ3_S, IQ4_XS, BF16, F32 | ADMITTED (every type is in `DequantGgufRowToBf16`) |
| `unsloth-flint-GGUF/flint-preview-UD-IQ1_M.gguf` (11 GB, sha256 `dd45136a...`) | IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ3_XXS, Q3_K, Q5_K, Q6_K, BF16, F32 | ADMITTED (all in the dequant set) |
| `flint-APEX-GGUF/Flint-APEX-{I-Mini,I-Compact,I-Quality,Compact,Quality,I-Balanced,Balanced}.gguf` | Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ4_XS, BF16, F32 | ADMITTED (all in the dequant set) |

Every ggml type the containers carry is already in
`src/vllm/model_executor/model_loader/gguf_dequant.cpp`'s capability set
(F32/F16/BF16/Q4_0/Q5_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ1_S/IQ1_M/IQ2_S/
IQ2_XS/IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS/MXFP4), so NOTHING is refused for
quant type — a refusal-by-name list is recorded here as empty ON PURPOSE,
because the kolibri1 precedent made an implicit refusal the defect. The loader
still refuses an unknown ggml type by name through `DequantGgufRowToBf16`'s
own missing-case error.

## Design

1. **Config from GGUF (ROUTE A, the kolibri1 shape).** `FlintHfConfigFromGguf`
   reads the file's own KVs and synthesizes the HF-shaped `HfConfig`, handing
   it to the SAME `ParseFlintParams` a config.json descends through. The layer
   schedule comes from the file's own `attention.sliding_window_pattern`
   (true = sliding); `attention.rope_pattern` must EQUAL it on these
   containers and the mismatch is a loud failure, which also re-derives
   `no_rope_layer_indices` exactly the way the parser validates it. Read and
   REQUIRED, not ignored: `expert_gating_func == 2` (sigmoid — the value the
   architecture defines; anything else is refused by name),
   `expert_weights_norm == true` (the renormalisation the forward always
   applies), `expert_weights_scale == 1.0`,
   `leading_dense_block_count` → `mlp_only_layers`.
2. **Weights into the EXISTING `FlintWeights`.** `LoadFlintFromGguf` mirrors
   `LoadFlintWeights`: every projection lands bf16 in the `[K=in, N=out]`
   Matmul-B orientation. GGUF rows come out `[out, in]`, so every 2-D
   projection is dequantized and TRANSPOSED (the kolibri1 loader needed no
   transpose because its forward consumed `[out, in]`; flint's does not).
   Expert towers dequantize ONCE per layer and slice per expert, then
   transpose, exactly the `OwnBf16RowSlice` cost shape. Norms and the sinks
   (`attn_sinks.weight`, F32 `[num_heads]`) convert to bf16. Full structural
   accounting both directions (missing = refuse at `GgufFile::Get` by name;
   unaccounted = refuse with the first name).
3. **Tokenizer.** `tokenizer.ggml.pre = "flint"` maps to a NEW
   `SplitPattern::kFlint` running the PR #30928 regex set as three stages:
   `\p{N}{1,3}`, the CJK range class, and the llama3/GPT-4o-family word rule
   (camel-split major/minor letter runs, case-sensitive contraction suffix,
   `[\p{P}\p{S}]` punct run with no `/` tail). Not aliased: the punct class
   excludes marks where `[^\s\p{L}\p{N}]` does not, and the slash tail is
   absent — both are real splitting differences against `kGpt4o`.
4. **The gate enters through the production entry point.** The test resolves
   `ModelRegistry::Resolve(HfConfig)`, loads with
   `ModelRegistry::Load(config, ModelSource::FromGguf(...))`, and forwards
   through `ModelRegistry::Forward` — never by constructing `FlintWeights`
   by hand. The dispatch reachability case in `tests/vllm/
   test_model_loader_gguf.cpp` proves `general.architecture = flint` reaches
   THIS family's builder and no longer the unsupported-arch refusal.

## Gates

- **The token gate.** Golden greedy chains from the pinned oracle
  (`llama serve` on the same container, temperature 0, top_k 1, raw prompt
  ids, no BOS — `tokens_evaluated == len(ids)` proves it) captured by
  `scripts/gen-flint-goldens.py` into `tests/vllm/models/flint_goldens_q4_k_m.json`,
  replayed through the production registry on the same container. Band: a
  divergent step is adjudicated by OUR second-best logit gap (the kolibri1
  contract verbatim); the Q4_K_M band is 2.5 nats, the same measured envelope
  W1's synthetic gate and the kolibri1 W3 gate use, NOT re-derived.
- **Config mapping** asserted against the file's own numbers (49 layers,
  hidden 2048, 32/4 heads, head_dim 128, ff 6144, 256/6 experts + shared
  1024, dense 0-1, window 513, FFSSSF).
- **Regression:** the W1 synthetic gate (`ctest -R flint`) stays green.

## Risks

- A transpose slipped on ONE projection class (dense vs routed vs shared vs
  router) produces plausible-but-wrong text; the greedy chains over eight
  prompts make that a loud failure, not a silent one.
- The bf16 expansion of a 40 B-param container is ~80 GB resident; the gate
  box has 255 GB and the load is one-shot.
- The oracle is an unmerged PR head; its pin file carries the retirement
  clause.

## Stop conditions

A hard flip (gap > band) on ANY prompt stops the row: no band widening. The
failure gets a layer-level diagnostic (per-layer logit dump vs the oracle's
`tensors` dump) before any other change.

## Owed

- The APEX/UD arm gates: this row gates the Q4_K_M container end to end; the
  other admitted containers load through the same code path but carry no
  captured goldens of their own in this wave.
- The paged/engine arm and the speed axis: unchanged from W1's `## Owed`.
- The roadmap has no `MODEL-TEXT-FLINT-GGUF` row; the work is filed under
  `MODEL-TEXT-FLINT` (whose `## Owed` names it). The roadmap row is the
  operator's to add.
