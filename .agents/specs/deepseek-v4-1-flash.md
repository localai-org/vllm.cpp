# DeepSeek-V4.1-Flash — scope the three published artifacts, and name what blocks each

Row: `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm` (primary), with
`MODEL-SPEC-deepseek-v4-1-dspark-v41-draft-model`, `MODEL-DSV41-EXL3` and `MODEL-DSV41-GGUF-Q1_0`.
Issues, one per row, all canonical-local:
`ISSUE-LOCAL-01M29ASR5N5AZ6YWMHFSR2WYCP` (primary),
`ISSUE-LOCAL-01M29AT0818WGXFSS4E03TT3KA` (DSpark draft),
`ISSUE-LOCAL-01M29ATB6N2SFZD7JA6CCR2KXC` (EXL3),
`ISSUE-LOCAL-01M29ATQDVEQ7M2MP3VW79XQJP` (GGUF Q1_0), and
`ISSUE-LOCAL-01M29AV2VX9RBRF409F6YVX3MK` on `QUANT-GGUF-Q1_0` for the reader
correction this scope uncovered.
Base SHA: `a2532471b`
Matrices: [model-matrix.md](../model-matrix.md) (the two registry arches, the
DSpark draft row, and the two checkpoint campaigns below),
[quantization-matrix.md](../quantization-matrix.md) (`QUANT-GGUF-Q1_0`).
**CORRECTED 2026-09-27: the two checkpoint campaigns are in `model-matrix.md`,
not `kernel-matrix.md`,** and this line used to say otherwise. Two rows decided
it. `MODEL-SPEC-deepseek-v4-1-dspark-v41-draft-model` — this spec's own
sibling row, same wave, same 2026-09-11 scoping pass — is declared in
`model-matrix.md`, as is `MODEL-DSV4-EXL3`, the V4 campaign row these two are
the successors of. `kernel-matrix.md` also pins its row count in two places
(`## Count invariants`: "exactly 52 practical kernel-family rows" and the
lifecycle tally, with `scripts/check-agent-record.py` holding the total), so
neither of these two rows can be added there without a second change to that
contract — which is a claim about cost, not a reason to file a model campaign
as a kernel family. Both rows were named by this header and by an open
canonical-local issue while appearing in NO matrix, so their issue records
failed `validate_issue_record` with "row is not canonical and claimable" and
`agent-issue-index.py` could not validate a single record.

This spec landed as records only and carries the campaign's product code from
W1 onwards; no row here is `READY`. The header names **five** rows. Of the four
V4.1 rows, `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm` is `PARTIAL`
since W1 landed on 2026-09-13 and the other three are `BLOCKED`; the fifth,
`QUANT-GGUF-Q1_0`, is not a V4.1 row at all — it stays `INVENTORIED`, as
`## Owed` records, because this scope only corrects the evidence on it. This
document says what blocks each of the four.

## Now

One row is `PARTIAL` and three are `BLOCKED`, and **not one of the four is
blocked on a single thing**. Three take the vLLM-pin blocker, one does not, and
every one of the four carries at least one further blocker that no pin advance
touches:

- `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm` — **`PARTIAL` since
  2026-09-13**, and still blocked for its GATE on the **pin**, AND hardware.
  W1 landed: the architecture RESOLVES through `ModelRegistry` and the real
  published `config.json` PARSES and VALIDATES; the loader, the forward, the
  KV-cache spec and the `deepseek41` GGUF container each refuse BY NAME. It
  does not load and does not forward. That moved the row off `BLOCKED` because
  `BLOCKED` bounds the gate rather than the work (below), and W1 claims no gate:
  registration and config validation produce no denominator. **Both gate
  blockers stand unchanged.** The pin: `DeepseekV41ForCausalLM` does not exist
  at `e126687a9a`, so there is no primary oracle to run W2 onwards against. The
  hardware: the release is 475.27 GiB against `dgx:gpu0`'s 119 GiB (3.99x) and
  neither quantized arm rescues it; `## Scope`'s table answers
  "Fits one GB10 (119 GiB)?" with `no` for the release, and
  `model-matrix.md:159` and `:547` carry the same. `## Work breakdown` W2-W10
  is what remains.
- `MODEL-SPEC-deepseek-v4-1-dspark-v41-draft-model` — the **pin**, AND the
  target row. A draft head is not portable before the model it drafts for
  (`model-matrix.md:160` and `:586`).
- `MODEL-DSV41-EXL3` — the **pin**, AND a 428.488 GiB 4-way tensor-parallel
  checkpoint that needs four MATCHED fleet devices the fleet does not have
  (below, and `## Dependencies`).
- `MODEL-DSV41-GGUF-Q1_0` — **not the pin at all**, and two blockers of its own:
  no released engine loads `deepseek41`, and no published rung both fits 119 GiB
  and keeps `token_embd`/`output` out of the 1-bit format (`## Dependencies`,
  and `kernel-matrix.md:162`, which leads with the no-released-engine half).

So a pin advance removes exactly one blocker — the **oracle**, and only on the
three rows that take it. It leaves every gate here blocked: on hardware, on
ordering, and on the published artifacts themselves. `## Scope`, `## Gates` and
`## Dependencies` carry the evidence for each; this section only says which row
takes which.

**`BLOCKED` bounds the GATE, not the work, and `## Work breakdown` says which
waves proceed anyway. W1 is that claim, executed.** The primary row's move to
`PARTIAL` is not a blocker lifting; it is this paragraph being true. An earlier
revision of that section put W1 "after a pin
advance", and the tree falsifies that: `Qwen3_5ForCausalLM` and
`Qwen3_5MoeForCausalLM` sit at `PARTIAL` today as an explicit **"Ahead-of-pin
forward port … REGISTERED, NOT RUN-GATED"**
([model-matrix.md](../model-matrix.md), issue
[#490](https://github.com/mudler/vllm.cpp/issues/490)), and `glm5_next` was
registered while vLLM implemented it at no revision at all. Registration, config
validation, the host op references and their device ports are all reachable
without the pin, because none of them claims a gate against the oracle. What the
pin gates is the DENOMINATOR, and nothing below produces one.

The pin blocker, taken by the first three rows, is this. `DeepseekV41ForCausalLM`
does not exist at our parity pin `e126687a9a` and cannot be made to exist there:
the architecture landed on vLLM `main` on 10-11 September 2026, and the commit
that completes it, `e77daef89e`, is **566 commits** past the pin (measured
`git rev-list --count e126687a9a..e77daef89e`). Until the pin advances there is
no primary oracle, and under
AGENTS.md no secondary oracle substitutes for a path vLLM implements.

`MODEL-DSV41-GGUF-Q1_0`'s second blocker, the one the bullet above states in a
line, is the artifact itself and no pin advance touches it either. The only
published rung that fits one GB10 (98.591 GiB against 119) stores `token_embd`
in a 1-bit format and is therefore **degenerate by construction** — it emits the
same token for every prompt, its producer root-caused it as the file rather than
the runtime, and the first rung with enough magnitude left, Q2_K, is 246.349 GiB.
No published rung is both loadable-and-meaningful and small enough for this box.

`MODEL-DSV41-EXL3` carries **both** the pin blocker and one more that no pin
advance touches. Its model half is the same unregistered `deepseek_v41`, and its
format half has no vLLM implementation at any revision, so it leans on the
secondary [`exllamav3`](../oracles/exllamav3.md) pin for the trellis bytes only.
On top of that the checkpoint is **428.488 GiB** over 114 files (460,085,046,068
bytes, re-derived 2026-09-11 from the HF tree API; this is the 428.49 GiB the
`## Scope` table and `kernel-matrix.md` round to, not a second figure) against
`dgx:gpu0`'s 119 GiB,
and it is a 4-way tensor-parallel artifact: running it needs four MATCHED fleet
devices, and the fleet has no four of one class. **The membership is what the
controller reports, not what AGENTS.md enumerates**, whose three names are
explicitly "a lower bound and never an upper one": `rc devices` run 2026-09-12
returns **four** devices — `dgx:gpu0` (state `unhealthy (no contact 25m8s)` at
that read), `orin:gpu0`, `strix:gpu0` and `thor:gpu0`. The count is the only
thing that correction moves. Four boxes of four DIFFERENT classes, one of them
unhealthy, is still not a 4-device tensor-parallel group, so the blocker stands
unchanged. `## Dependencies` carries it in full.

## Scope

Three published artifacts for one model, `deepseek-ai/DeepSeek-V4.1-Flash`
(552B backbone, 8B activated at prefill and 16B at decode, MIT):

| Artifact | Revision | Size | Fits one GB10 (119 GiB)? |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V4.1-Flash` (bf16 + fp8 blocks, fp4 experts) | `dba1be0a40aa45a94ad051997016db3960a90277` | 475.27 GiB | no |
| `bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard` | `f129e31a81e1337aa33e129e2d847fc7e37c8733` | 428.49 GiB | no — a 4x DGX Spark TP4 checkpoint |
| `vcruz305/DeepSeek-V4.1-Flash-GGUF`, `Q1_0` rung | `543d86fd975044332d4366204eadd6da1b0cbeda` | 98.59 GiB, 3 shards | **yes**, and it is the only one that does |

**Two denominators for the release, 27 MB apart, and both are used below on
purpose.** The **475.27 GiB** in the table is the whole repository tree at
`dba1be0a`: **510,313,353,565 bytes** over 95 entries, summed from the HF tree
API (`.../tree/dba1be0a...?recursive=1&expand=1`, LFS sizes, paged by the `Link`
header, re-derived 2026-09-11), so it includes the tech-report PDF, the assets
and the reference `encoding/` sources. The **475.24 GiB** the engram measurement
in `## Risks/decisions` divides by is `model.safetensors.index.json`'s own
`metadata.total_size`, **510,286,023,000 bytes** — tensor bytes only, which is
the right denominator for "what fraction of the WEIGHTS is engram". The gap is
27,330,565 bytes of non-tensor files, not a disagreement. (A third figure exists
and is deliberately not used: the 48 `.safetensors` files alone sum to
510,296,708,312 bytes = 475.25 GiB, tensor data plus per-shard headers.)

In scope: the records that make the gap legible — the two registry rows, the
two checkpoint campaigns, and the `QUANT-GGUF-Q1_0` evidence correction.
Out of scope: any implementation, any pin advance, any download, any lease.

### The developer's ask, answered against the artifacts

The direction given in session on 2026-09-11 was to scope the three artifacts,
and, if the model is too big, that **the Q1_0 version should run at least on
the dgx**. Measured against the published files, the answer is that **this particular file
cannot satisfy it, and no runtime work would change that.** The obstacle is not
size, and it is not the loader.

- **Size is satisfied.** Measured from the HF tree API on 2026-09-11: the three
  shards are 38.781 + 41.776 + 18.033 = **98.591 GiB**, leaving about 20.4 GiB
  of `dgx:gpu0`'s 119 GiB unified pool for KV and activations.
- **The Q1_0 rung is degenerate BY CONSTRUCTION, and the producer root-caused
  it.** On ggml-org/llama.cpp#28696 (read 2026-09-11) the author writes that it
  "executes, and it emits the same token for every prompt. That is the file, not
  the runtime." The mechanism is the one that matters here: a 1-bit block format
  keeps one scale per block and one sign bit per weight, and this file stores
  **`token_embd` itself at type 41**, so the embedding table comes back as plus
  or minus a single magnitude. "The embedding is the only place prompt identity
  enters the model", so once it is a sign vector the prompts are **nearly**
  indistinguishable — the author's qualifier, kept — and the 40 equally
  quantized layers finish the job. The
  author measures it at **1.53 bits per weight over the backbone**, 1.13
  counting the engram tables, and states that **Q2_K is the first rung with
  enough magnitude left**.
- **That mechanism is the author's, and it is not the only Q1_0-specific defect
  he names.** In the same comment thread he also fixed `llama-quant.cpp`, which
  "picks tensors by name, so it was quantizing two per-channel scale vectors
  that are applied with `ggml_mul`, which has no quantized path. At Q1_0 a scale
  vector becomes plus or minus one scale, which is not a scale." So read the
  embedding mechanism as the one he root-caused, not as the sole cause proven to
  the exclusion of the other. The **published** file appears to postdate that
  second fix and is not evidence for it: over an HTTP range read of shard 1 on
  2026-09-11 its 28 `blk.N.hc_{attn,ffn}_scale.weight` tensors are all `F32`,
  and its type histogram is 204 Q1_0, 164 F32, 14 BF16 and one Q6_K
  (`output.weight`), with `token_embd.weight` Q1_0.
- **Q2_K does not fit.** It is 246.349 GiB, a little over twice the pool. So
  between "fits the box" and "can say anything", the published rungs have no
  overlap, and that is the finding this row exists to record.
- **The loader blocker is already gone, and the card is stale.** As of
  2026-09-11 the published files carry the nine engram keys through a KV-only
  repair, tensor payload SHA-256 identical to the originals, and the producer
  states they should load as they are. The sparse attention is also NOT missing:
  `f0159e2` is the parent of the tested `f37da5711`, so `build_attention_v41`
  with the shared compressed stream, shared index keys and indexer top-k was
  already present. Do not carry the model card's "sparse attention outstanding"
  into any later decision.
- **`deepseek41` is still not in a released engine.** It is absent from
  llama.cpp `b10451` ([our pin](../oracles/llama-cpp.md)) and from its `master`;
  #28696 is a DRAFT and conversion-only; the runtime is the out-of-tree branch
  `vcruz305/llama.cpp` `runtime/deepseek41`. And `vllm-gguf-plugin` at our pin
  ([oracle](../oracles/vllm-gguf-plugin.md) `d4c1f0d082`) has no DeepSeek
  adapter at all — V4 is open at its #114 and #126, V4.1 is nowhere.

So the Q1_0 arm is rowed and `BLOCKED`, and the row records the mechanism rather
than the symptom, because the mechanism says what a fix would have to be. A rung
that fits this box has to keep `token_embd` (and `output`) out of the 1-bit
format, the way this same file already keeps `output.weight` at Q6_K and the
router at BF16. That is a requantization question, not a runtime question, and
it is the only identified path to the developer's ask on this hardware.

### What is NOT a blocker, and was checked because it looked like one

`Q1_0` is ggml type id **41**, and this tree used to call that a private
extension of mudler's killgate llama.cpp fork. That is now false and the
difference decides whether the 98.59 GiB file is readable here at all.

- Stock ggml-org/llama.cpp at our pinned tag `b10451`
  (`10bf611e533d81f739128304991c5e133c6aebd8`) defines
  `GGML_TYPE_Q1_0 = 41` (`ggml/include/ggml.h:431`, with `Q2_0 = 42` on `:432`
  and `GGML_FTYPE_MOSTLY_Q1_0 = 27` on `:476`) with
  `QK1_0 = 128` and `block_q1_0 = { ggml_half d; uint8_t qs[16]; }`
  (`ggml/src/ggml-common.h:180-185`) = **128 elements / 18 bytes**, 1.125 bpw,
  and `gguf-py/gguf/constants.py:5437` agrees. It was added by merged PR
  ggml-org/llama.cpp#21273 (2026-04-06). `master` is byte-identical here.
- The killgate fork at `237ad9b96` and the producer's fork
  `vcruz305/llama.cpp` both carry the same id at the same geometry.
- The artifact agrees. Over an HTTP range read of
  `DeepSeek-V4.1-Flash-Q1_0-00001-of-00003.gguf`, consecutive tensor-offset
  spans give exactly 18/128 = 0.140625 bytes per element; for example
  `blk.0.ffn_down_exps.weight` `[2304,5120,384]` = 4,529,848,320 elements in
  637,009,920 bytes.

Our reader's traits row is therefore CORRECT:
[`gguf_reader.cpp:387-391`](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L387-L391)
gives id 41 as `{128, 18, "Q1_0"}` — `case 41:` is the label at `:387` and the
`static constexpr GgmlTypeTraits t{128, 18, "Q1_0"};` literal is at `:391`. What
is stale is the provenance, and it is stale at **three** comment sites and not
one: the header comment at
[`:194`](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L194), and
the two per-case comments that each open "Killgate fork extension:" —
[`:379`](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L379)
(`block_nvfp4`, id 40) and
[`:388`](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L388)
(`block_q1_0`, id 41). The last of those sits INSIDE the `:387-391` range cited
above, whose traits literal is nonetheless correct: the numbers are right and the
attribution above them is not. Mainline additionally defines
`GGML_TYPE_Q2_0 = 42`, which `FindGgmlTraits` does not handle and silently falls
through to `nullptr`. All of it is owed below; none of it blocks this scope.

## Upstream chain

vLLM `main` implements V4.1 as a **fork of the V4 tree, not an extension of
it**. Read first-hand at `e77daef89e` in the checkout `.env`'s `VLLM_SOURCE`
names: `git ls-tree` gives `vllm/models/deepseek_v4_1/` as `__init__.py`,
`attention.py`, `compressor.py`, `quant_config.py`, `sparse_mla.py`, `amd/`,
`common/` and `nvidia/` — duplicating the V4 modules, subclassing V4
selectively, and shipping **no `cpu/`, no `xpu/` and no `mtp.py`**. Three
merged pull requests:

| PR | Commit | Date | Scope |
|---|---|---|---|
| vllm#56228 | `9b959b86577c082c0b2bf9e2c22263255a36ad83` | 2026-09-10 | model definitions, +12902/-196 over 47 files |
| vllm#56208 | `b47b01cf3383d50a5fae8569cadcbd0736076234` | 2026-09-10 | config class, tokenizer, renderer, reasoning and tool parsers |
| vllm#56214 | `e77daef89e18e08321ae7b8b24827eedd5fe8673` | 2026-09-11 | engine wiring: registry, KV cache, indexer backend, spec-decode, warmup |

Registry at `e77daef89e`, read first-hand:
`model_executor/models/registry.py:379` `"DeepseekV41ForCausalLM"` and `:647`
`"DSparkV41DraftModel"`; config `transformers_utils/config.py:156`
`**{"deepseek_v41": "DeepseekV41Config"}`.

**The negative at our pin is measured, not assumed.** At `e126687a9a`,
`grep -c` for `DeepseekV41|deepseek_v4_1` in `registry.py` is **0** and for
`deepseek_v41` in `transformers_utils/config.py` is **0**; `vllm/models/` holds
`deepseek_v32` and `deepseek_v4` and no `deepseek_v4_1`; `registry.py:98`
registers `DeepseekV4ForCausalLM` and `:624` `DSparkDraftModel`. So this is
ABSENCE at the pin and PRESENCE upstream — the one shape a pin advance, and
only a pin advance, can resolve.

**The architecture the model is named for is a Causal Encoder-Decoder, and the
first revision of this spec never said so.** The release card
(`deepseek-ai/DeepSeek-V4.1-Flash` `README.md:47-49`, read 2026-09-11) states
that V4.1-Flash "adopts a **Causal Encoder-Decoder (CED)** architecture: a
40-layer Transformer organized as a 20-layer causal encoder followed by a
20-layer decoder", in which "the decoder's global KV cache is projected from the
final encoder hidden states rather than derived from each decoder layer's own
hidden states" — which is where the 8B-at-prefill against 16B-at-decode split
comes from. The card pairs it with **SWA Bounded Replay**, which "reconstructs
missing SWA KV states by replaying only the most recent *n*_win tokens", taking
the persistent KV footprint to roughly 1/8 of V4-Flash, and with **CSA2**, which
gives each attention layer one of three static modes — Full, Reindex or Reuse.

**The 20/20 boundary is visible in the config; the two source-layer lists are
NOT the mechanism that implements it.** `text_config.compress_ratios` at
`dba1be0a` reads `0, 0` then eighteen `2`s for layers 0-19, twenty `1`s for
layers 20-39, and three trailing `0`s for the MTP layers — a clean split at
layer 20, which is also `candidate_source_layer_id`. But
`kv_source_layer_ids = [2, 8, 14, 20]` and
`index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]` encode CSA2's
CROSS-LAYER SHARING, and both references say so in nearly the same words. The
release's own `inference/model.py:82-84` comments them as "layers sharing a
ratio also share one compressed KV and one indexer, produced by the first", and
vLLM's `deepseek_v4_1/attention.py:244-292` at `e77daef89e` as "Compressors and
compressed-KV caches live only on `kv_source_layer_ids`; indexers only on
`index_source_layer_ids`. Consumers reuse the most recently published source
below them", resolving each consumer as
`max(s for s in self.kv_source_layers if s <= layer_id)`. A reader who takes
those two keys for the CED projection ports the wrong thing.

**UNVERIFIED: where the CED projection and the replay live in code.** Neither
the released `inference/model.py` nor vLLM's `deepseek_v4_1/` package uses the
words encoder, decoder or replay for this at all — `grep -ci encoder` over
`inference/model.py` is 0. Over vLLM's V4.1 package, re-run 2026-09-11,
`git grep -in encoder e77daef89e -- vllm/models/deepseek_v4_1/` returns 9 lines
and `git grep -in decoder e77daef89e -- vllm/models/deepseek_v4_1/` returns 24,
so the count is not the evidence and what those lines ARE is. None of the 33
names the Causal Encoder-Decoder split. The 9 encoder lines are the vision
path's `supports_encoder_tp_data` flag (`nvidia/vl_model.py:127`,
`amd/vl_model.py:127`), the multimodal `EncoderCache` import and the constructor
argument it is passed as (`nvidia/model_state.py:11,58,61`), and four fp8 cache
comments in which "encoder" means the kernel that WRITES a cache
(`amd/rocm.py:850`, `common/ops/cache_utils.py:411,412,413`). All 24 decoder
lines are the ordinary per-layer block type `DeepseekV4DecoderLayer`: its
definition (`nvidia/model.py:163`, `amd/model.py:156`), its construction and
`isinstance` checks, its reuse by the dspark draft (`nvidia/dspark.py:54,114`,
`amd/dspark.py:51,110,223`), and comments about decoder-layer weights and
ordering. `git grep -in replay` over the same package returns 3, and all three
are CUDA GRAPH replay (`nvidia/flashmla.py:229`,
`nvidia/model_state.py:65,95`). The card and the
`compress_ratios` boundary are this spec's whole evidence for CED and for
Bounded Replay. Do not port either from this document.

The six subsystems with no counterpart in this tree:

- **The CED encoder-decoder split and SWA Bounded Replay** — nothing here
  divides a stack into a KV-producing encoder half and a consuming decoder half,
  and nothing reconstructs a sliding-window KV state by replaying tokens. Our V4
  path gives every layer its own KV. Marked UNVERIFIED above at code level: the
  claim is the card's, and the only thing measured here is the 20/20
  `compress_ratios` boundary.
- **Engram** — `deepseek_v4_1/common/engram.py`, 1033 lines at `e77daef89e`:
  n-gram hash
  lookups gated into the hyper-connection stream, two embedding tables of
  384,006,168 and 384,016,682 rows, **39.8% of the checkpoint's bytes**
  (measured, below), carrying
  cross-forward state (a token to compressed-vocab map plus an int32 hash-cache
  slot per KV slot, 3-token lookback in a 128 sliding window). New engine config
  object `vllm/config/engram.py:38`, resolved at `vllm/config/vllm.py:1152-1169`.
- **MXFP8 32x32 UE8M0 linears** — `deepseek_v4_1/quant_config.py:189` routes
  a `weight_block_size == [32, 32]` scale-e8m0 `LinearBase` to
  `ModelOptLinearMethod(kMxfp8Static / kMxfp8Dynamic)`. V4 used 128x128 fp8.
  **What V4.1 adds here is the ROUTING, not the kernel directory.** An earlier
  revision of this bullet said "New kernel directory
  `model_executor/kernels/linear/mxfp8/`", and that is false: the directory
  already holds **9 files** at our pin
  (`git ls-tree -r e126687a9a -- vllm/model_executor/kernels/linear/mxfp8/`,
  run 2026-09-11: `Mxfp8LinearKernel.py`, `__init__.py`, `b12x.py`,
  `emulation.py`, `flashinfer.py`, `humming.py`, `marlin.py`,
  `rocm_native.py`, `xpu.py`). Over the range to `e77daef89e`,
  `git diff --name-status e126687a9a e77daef89e -- <that path>` adds exactly
  **one** file, `deep_gemm.py`, and modifies `Mxfp8LinearKernel.py` and
  `flashinfer.py`. So the V4.1 delta is the `[32, 32]` / `is_scale_e8m0`
  branch at `deepseek_v4_1/quant_config.py:189` plus that one new kernel
  backend, against a mxfp8 stack we already have a pin-side reading of.
- **A second indexer block size** — `DeepseekV41IndexerBackend` at
  `v1/attention/backends/mla/indexer.py:252` (64/128-token blocks against V4's
  256), with the ops `indexer_k_store.py` and `query_quant.py`.
  **The three sibling "new" files in this list ARE new, re-verified 2026-09-11,
  and they are new in two different ways.** `vllm/config/engram.py` is absent at
  `e126687a9a` and present at `e77daef89e` (`git cat-file -e <rev>:<path>`), so
  it is a genuinely new file in an existing directory. The two ops are
  `vllm/models/deepseek_v4_1/common/ops/indexer_k_store.py` and
  `.../query_quant.py`, and they are new because their whole package is:
  `git ls-tree -r e126687a9a -- vllm/models/deepseek_v4_1/` returns **0**
  entries, so every path under `deepseek_v4_1/` this spec cites arrives with
  the architecture rather than being edited into place.
- **DSpark-only drafting** — `config/speculative.py:682-687,1347,1422-1426`
  branch on `deepseek_v41`; there is no `DeepseekV41MTP` anywhere on `main`.
- **A vision entry point** — `DeepseekV41ForCausalLM` is the multimodal class
  (`deepseek_v4_1/nvidia/vl_model.py:120`), the text-only class being
  `DeepseekV41LLMForCausalLM` (`nvidia/model.py:979`).

Config deltas that a loader sees first, `deepseek-ai/DeepSeek-V4.1-Flash`
`config.json` at `dba1be0a` against `deepseek-ai/DeepSeek-V4-Flash` at
`60d8d70770`: nested `text_config` + `vision_config` where V4 was flat;
`DeepseekV41ForCausalLM` / `deepseek_v41`; hidden 4096 to **5120**; layers 43 to
**40**; routed experts 256 to **384**; `moe_intermediate_size` 2048 to **2304**;
`q_lora_rank` 1024 to **1280**; `index_n_heads` 64 to **32**;
`rms_norm_eps` 1e-6 to **1e-20**; `weight_block_size` [128,128] to **[32,32]**;
`num_nextn_predict_layers` 1 to **3**; `compress_ratios` from 44 alternating
entries to 43 blocked ones; `num_hash_layers` REMOVED, and with it V4's
hash-routing; `engram_*` (8 keys), `dspark_*` (6 keys), `vision_config` (**11**
keys counted from the live `config.json`, `model_type:
deepseek_v41_vision` among them), `kv_source_layer_ids`,
`index_source_layer_ids`, `candidate_*` all NEW — the last three being CSA2
sharing and candidate pre-filtering, not the CED projection, as above.
Unchanged and therefore reusable: head_dim 512, `qk_rope_head_dim` 64,
`o_lora_rank` 1024, `o_groups` 8, 64 heads over 1 KV head, vocab 129280,
`sliding_window` 128, `swiglu_limit` 10.0, `scoring_func` `sqrtsoftplus`,
`topk_method` `noaux_tc`, `routed_scaling_factor` 1.5, `index_head_dim` 128,
`index_topk` 512, the three `hc_*` values, and `compress_rope_theta` 160000.

Secondary references, for the paths vLLM's registration does not reach:
[`exllamav3`](../oracles/exllamav3.md) for the EXL3 trellis bytes, and
[`llama-cpp`](../oracles/llama-cpp.md) for the GGUF encodings. Neither outranks
vLLM, and neither implements V4.1.

## Our baseline

DeepSeek-V4 is `ACTIVE` in this tree and none of it transfers unmodified. The
V4 forward, its four host primitive families and their CUDA ports live in
`deepseek_v4*.{h,cpp,cu}`; the GGUF arm maps the **`deepseek4`** architecture
only, at
[`model_loader.cpp:1243`](../../src/vllm/entrypoints/model_loader.cpp#L1243)
(`{"deepseek4", &vllm::DeepseekV4HfConfigFromGguf}`), so a `deepseek41` file is
refused by name today. `MODEL-DSV4-EXL3` carries the EXL3 loader and kernels
against a V4 checkpoint. `QUANT-GGUF-Q1_0` is `INVENTORIED`: the reader
recognises the type and no executable path consumes it.

What V4.1 can reuse, if and when it is implemented: the 512-wide MLA output
seams, the sink softmax and grouped output-LoRA, the DSA indexer and
compressor, the MHC Sinkhorn family, the `sqrtsoftplus` router and clamped
SwiGLU, and the EXL3 trellis dequant. What it cannot: engram, MXFP8 32x32,
the second indexer block size, DSpark, the vision tower, and the CED
encoder-decoder KV split with its SWA Bounded Replay.

**What CED means for reuse of the V4 attention path.** Our V4 block has one
polarity: each layer produces the KV it consumes. V4.1 breaks that twice over,
and the two breaks are different sizes. The CSA2 sharing is the smaller one and
is already visible in code on both sides — a consumer layer reads the compressed
KV and index keys published by the nearest source layer at or below it, so a
port needs a per-layer source resolution and a KV allocation that is keyed by
source layer rather than by layer. That is a plumbing change to `AttnBlock`, not
new numerics. The CED projection is the larger one, because the decoder's global
KV is described as a function of the FINAL encoder hidden state rather than of
each decoder layer's input, which no existing schedule here can express and
which nothing in this spec has read in code. Neither break is scoped; both are
recorded so a later port does not discover the polarity late.

## Port map

Not applicable in this change and deliberately so — no product code lands here.
When a port starts, it lands as new files mirroring the upstream fork
(`deepseek_v4_1*`), never as a widened `deepseek_v4`, because upstream itself
chose duplication over extension and mirroring that choice keeps the V4 rows
gateable while V4.1 moves.

## Tests to port

None in this change. Recorded so the eventual port cannot skip them: vLLM's
V4.1 suites, the engram hash-state tests, and the MXFP8 32x32 scale tests are
the ports owed, each with its upstream revision anchor, in the change that
first makes the corresponding path executable.

## Gates

No gate can run at this revision, and this section records why rather than
leaving it to be discovered.

- **Token-exact vs pinned vLLM: IMPOSSIBLE at `e126687a9a`.** The pinned oracle
  cannot parse a V4.1 `config.json`; it has no `deepseek_v41` model type and no
  flattener for the nested config shape.
- **Secondary oracle: FORECLOSED.** vLLM implements the behaviour on `main`, so
  under AGENTS.md SGLang, transformers and llama.cpp are not behaviour sources
  for it.
- **The Q1_0 vehicle: NO DENOMINATOR, AND NOTHING TO MEASURE.** No released
  engine loads `deepseek41`, and the one rung that fits this box is degenerate
  by construction, so there is neither a reference to compare against nor an
  output worth comparing.

The gate these rows will take, once the pin advances: the declared token-exact
SACRED gate against pinned vLLM on identical artifacts, prompts and sampling,
before any speed axis is quoted.

## Dependencies

- A vLLM pin advance to `e77daef89e` or later. This is owned separately and is
  NOT assumed by this spec.
- The pin's own outstanding debt, which an advance would stack onto rather than
  clear: the step-6 benchmark re-baseline owed by #2818, which is OPEN
  (`gh issue view 2818 --repo mudler/vllm.cpp --json number,state,closedAt`
  gives `"state":"OPEN"`, `"closedAt":null`, read 2026-09-11). This bullet used
  to name a second debt, "the token-exact gate at `e126687a9a` owed by #2794".
  That is no longer true: the same command on 2794 gives `"state":"CLOSED"`,
  `"closedAt":"2026-09-06T07:00:58Z"`, so #2794 is discharged and does not size
  a pin advance. `.agents/upstream-sync.md` carried the stale sentence when this
  was written and was out of scope for this change; it was reconciled on
  2026-09-12 under row `ENG-RECORD-CLAIM-AGREEMENT` and no longer does.
  **That reconciliation also corrected this bullet's own framing:** #2818 is not
  the only debt the pin carries. The other four strict goldens at
  `e126687a9a` — 27B W4A4, 32B-NVFP4A16, 35B, Coder — are still owed too. They
  carry no issue by design, living under `## Owed` in
  `.agents/specs/upstream-sync-headpin-tokengate.md:569`, which is why an
  issue-state read alone cannot see them and why this bullet found only one
  debt.
- For the GGUF arm: an upstream or first-party `deepseek41` runtime, AND a rung
  that both fits 119 GiB and keeps `token_embd`/`output` out of the 1-bit
  format. Neither exists today, and the second one is a requantization we would
  have to produce, not something to wait for.
- For the EXL3 arm: 4 MATCHED fleet devices, which the fleet does not have as
  one tensor-parallel group. The membership is the controller's list and not
  AGENTS.md's three-name lower bound: `rc devices` run 2026-09-12 returns
  **four** — `dgx:gpu0` (`unhealthy (no contact 25m8s)` at that read),
  `orin:gpu0`, `strix:gpu0` and `thor:gpu0`. Four boxes of four different
  classes, one unhealthy, is not a tensor-parallel group, so only the count
  changes here and not the blocker.
- **V4.1 itself changes no requirement at all, and the advance that would deliver
  it changes 15 files.** The three V4.1 commits touch `requirements/` in ZERO
  files (`git show --name-only --format= <sha> -- requirements/` is empty for
  each of `9b959b8657`, `b47b01cf33` and `e77daef89e`). Over the range,
  `git diff --name-only e126687a9a e77daef89e -- requirements/` lists **15**:
  `build/rock.txt`, `build/rocm.txt`, `common.txt`, `cuda.txt`, `rocm.txt`,
  `rubin-prerelease.txt`, `test/cpu.txt`, `test/cuda.in`, `test/cuda.txt`,
  `test/nightly-torch.txt`, `test/rocm.in`, `test/rocm.txt`, `test/xpu.in`,
  `test/xpu.txt` and `xpu.txt`. The notable moves are torch `2.11.0` to
  `2.12.0+rocm7.14.0` with matching torchvision/torchaudio/triton, transformers
  `5.15.0` to `5.16.1` with tokenizers `0.22.2` to `0.23.1`, `mcp` and `mcp-types`
  **converging on `2.1.1` across four files carrying three distinct starting
  values -- `1.28.1` on cpu and cuda, `1.27.0` on rocm, `2.0.0` on xpu -- one of
  which was already 2.x** (two earlier revisions of this bullet were narrower:
  the first said "`mcp` `1.28.1` to `2.1.1` (and the new `mcp-types`)", which is
  the cpu/cuda story read as if it were the whole one; the second said "four
  different starting points", which counted the FILES and not the VALUES, while
  the enumeration below it was already exact; the three-value count was
  re-derived 2026-09-12 with the same command). Re-derived 2026-09-11 with
  `git show <rev>:requirements/test/<f>.txt | grep -i '^mcp'`: at `e126687a9a`
  `test/cpu.txt` and `test/cuda.txt` pin `mcp==1.28.1`, `test/rocm.txt` pins
  `mcp==1.27.0`, and `test/xpu.txt` ALREADY pins `mcp==2.0.0` **with
  `mcp-types==2.0.0`**; at `e77daef89e` all four pin `mcp==2.1.1` and
  `mcp-types==2.1.1`. So `mcp-types` is not new to the tree, only to
  cpu/cuda/rocm, and on xpu it is a 2.0.0 to 2.1.1 bump. `requirements/common.txt`
  moves from an unpinned bare `mcp` to `mcp >= 2.0.0, < 3.0.0` in the same range,
  which is where the major-version floor actually lands. Also
  `openai >= 2.0.0` to `>= 2.25.0`, and
  FlashInfer `0.6.18` to `0.6.18.post1`. An earlier revision of this bullet said
  the FlashInfer bump was the only change and attributed it to V4.1; both halves
  were wrong, and the correction raises rather than lowers the cost this spec
  argues a pin advance carries.

## Work breakdown

**Developer goal, given 2026-09-12 in session:** land DeepSeek-V4.1-Flash
support, complete with the GPU op families, on Metal and on Vulkan as well as
CUDA. The waves below are that goal decomposed. Each is a row-sized brick with
its own fresh implementer and fresh reviewer, and each states what gates it.

**Two constraints bound every wave, and neither is removed by a pin advance.**

1. **No published artifact fits any device this project reaches**, so no
   end-to-end token gate against vLLM is reachable here. The release is 475.27
   GiB against `dgx:gpu0`'s 119 GiB; the EXL3 arm is 428.488 GiB and wants four
   matched devices; the only rung that fits, `Q1_0` at 98.591 GiB, is degenerate
   by construction. Every wave below is therefore gated the way V4's W3-W7 were
   gated: a portable host reference per op family, unit-gated at small shapes
   against hand-derived and double-precision references, and device ports gated
   against those host refs. That is a real gate and it is not an e2e gate, and
   this spec says so in both directions.
2. **DeepSeek-V4 itself has device ops on CUDA only** (`src/vt/cuda/cuda_deepseek_v4.cu`).
   `src/vt/metal/` and `src/vt/vulkan/` carry no DeepSeek family at all. So the
   Metal and Vulkan waves are not ports of an existing V4 arm; they are first
   implementations for this architecture on those backends.

| Wave | Scope | Needs the pin? | Needs a device? |
|---|---|---|---|
| W1 | **LANDED 2026-09-13.** Resolve and register `deepseek_v41`; parse and VALIDATE the nested config; refuse every unimplemented arm BY NAME. Moved the row to `PARTIAL`; closed `ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP` | no | no |
| W2 | Advance the vLLM parity pin to `e77daef89e` or later, reconciling every affected row and gate | it IS the pin | yes, for revalidation |
| W3a | **LANDED 2026-09-13.** **Engram** host reference: the n-gram hash state, the compressed token map, the two ~384M-row tables, and the gate into the hyper-connection stream. Landed UNREACHED, per `## Owed`, so `ISSUE-LOCAL-01M2C40RNXB871VW0E560PVBFA` stays OPEN | no | no |
| W3b | **LANDS WITH THIS CHANGE, 2026-09-13.** **MXFP8 32x32 UE8M0** host reference: block dequant and the linear arm, against V4's 128x128 fp8. Lands UNREACHED, per `## Owed` | no | no |
| W3c | **Indexer arm** host reference. The V4.1 backend delta is TWO methods (`indexer.py:252-259`): a name, and `get_supported_kernel_block_sizes()` returning 64 on Hopper and 128 elsewhere against V4's 256 -- arch-conditional, not a flat change. The net-new op is `indexer_k_norm_rope_store` (`indexer_k_store.py:29,120`), which replaces V4's indexer-local `DeepseekCompressor`: index K is derived from the MAIN compressor's latent as `k_norm(wk(latent))`, RoPE'd at the group's FIRST position `(pos//r)*r`, emitted only at group boundaries `(pos+1)%r==0`, into a 132-byte fp8 or 68-byte MXFP4 paged row with values and scales in separate regions. `fused_indexer_q_rope_quant` and `fused_inv_rope_fp8_quant` are NOT replaced -- `common/ops/__init__.py:12-22` re-exports them byte-identically from V4. `query_quant.py` is a CUDA+Blackwell-only fusion behind `can_fuse_query_quant`; a host port implements plain `fused_q_kv_rmsnorm` and is correct, so it is owed perf, not a gap | no | no |
| W3d | **The `kv_source_layer`-keyed KV topology** -- what the card calls CED and CSA2, which are ONE work item and not two. **There is no encoder-decoder split to port.** Measured 2026-09-13 at `e77daef89e`: `bounded.?replay\|csa2` returns ZERO across all of `vllm/`, all 40 layers are built by ONE `make_layers` call with one lambda (`nvidia/model.py:451-460`), and the 9 `encoder` / 24 `decoder` hits are vision-TP flags, multimodal `EncoderCache`, fp8 `encoder`-as-writer comments and the ordinary `DeepseekV4DecoderLayer`. The topology is CONFIG: `max(s for s in kv_source_layers if s <= layer_id)` (`attention.py:284-286`, index twin `:287-289`) resolves every layer 20-39 to layer 20, whose compressor consumes layer 19's output -- which IS the card's "decoder KV projected from the final encoder hidden states". `compress_ratios` flips 2 to 1 at layer 20 and `candidate_source_layer_id` is 20; three fields agree on one boundary. Only FOUR compressed-KV caches and FOUR indexer-K caches exist (layers 2/8/14/20). **SWA Bounded Replay is a persistence policy, not a forward-pass mechanism** -- each layer's 128-token ring is a pure function of that layer's own hidden states, so a correct forward owes nothing for it; whether vLLM implements it at all is UNVERIFIED and the reading is that it does not | no | no |
| W3e | **DSpark** host reference: block drafting at `dspark_block_size` 5 over target layers [37,38,39], its own 128-expert MoE at top-3, and the Markov rank-256 state | no | no |
| W3f | **Vision** host reference: the 32-layer DeepSeek-ViT, 2D RoPE, the 3x3 pixel-unshuffle and the 2-layer MLP projector | no | no |
| W4 | Host FORWARD assembly at a tiny synthetic config, composing W3a-W3f, finite and deterministic end to end | no | no |
| W5 | CUDA device ports of every W3 family, each gated against its host reference | no | yes (CUDA) |
| W6 | **Metal** ports of the same families, gated against the same host references | no | yes (the M4 host, which is NOT an `rc` fleet device and takes the file mutex) |
| W7 | **Vulkan** ports of the same families, gated against the same host references through `llvmpipe`, which is how the EXL3 Vulkan arm was proven byte-exact | no | no (llvmpipe is CPU) |
| W8 | Loader: the safetensors arm for the nested config, and the `deepseek41` GGUF architecture arm beside the existing `deepseek4` one | no | no |
| W9 | **A rung that both fits and speaks.** Requantize so `token_embd` and `output` stay out of the 1-bit format, the way the published file already keeps `output.weight` at Q6_K and the router at BF16. This is the ONLY identified path to running this model on this hardware at all | no | yes, to produce and verify |
| W10 | End-to-end run and whatever gate is then reachable | yes | yes |

**Sequencing.** W1 started immediately and blocked nothing else; it has landed. W3a-W3f are
mutually independent and are the parallel bulk of the campaign. W4 needs all of
W3. W5, W6 and W7 each need W4 for the composition gate but their per-family
ports need only the matching W3 family, so a family can go host-then-CUDA-then-
Metal-then-Vulkan without waiting for its siblings. W9 is independent of all of
it and is on the critical path for anything end to end. W10 needs W2 and W9.

**What "supported" will mean when these land.** The architecture resolves,
validates, loads, and forwards; every op family has a host reference and a CUDA,
Metal and Vulkan arm gated against it; every unimplemented path refuses by name.
It will NOT mean a token-exact e2e result against vLLM, unless W9 produces a
fitting non-degenerate artifact and a device is free to run it. Recording that
now is the difference between a campaign that ends in a measured result and one
that ends in a claim.

## Risks/decisions

- **Decision (developer, 2026-09-11, in session).** Record the rows as
  `BLOCKED` naming the pin advance, and keep the advance itself a separate
  owned decision. The alternative offered was to advance the pin inside this
  change; it was declined, and the reason stands on its own: the jump
  re-validates every model row and gate while the current pin's gate debt is
  unpaid. **The two distances are different and both are measured**
  (2026-09-11): `git rev-list --count e126687a9a..e77daef89e` = **566**, the
  distance to the V4.1 head this spec's `## Dependencies` names, and
  `git rev-list --count e126687a9a..origin/main` = **596** to `2d75e586fc`,
  today's `main`. 566 is the number that bounds this decision.
- **Decision (developer, 2026-09-11, in session).** Row the Q1_0 arm as
  `BLOCKED` with the degeneracy named, rather than starting a read path against
  an artifact no engine can gate.
- **Risk: this artifact's facts moved twice in one day.** Between the model
  card, the convert PR and its comment thread, the engram-key load failure was
  reported and fixed, the published files were rewritten in place, and the
  "sparse attention missing" claim was retracted by the author — all on
  2026-09-11, the day this spec was written. Every negative recorded here names
  what it was read from and when. Re-read before acting on any of them.
- **Decision NOT taken: requantizing our own rung.** The mechanism above says a
  fitting, non-degenerate rung is constructible — keep `token_embd` and `output`
  out of the 1-bit format, spend the ~20 GiB of headroom on them. That is a real
  path to the developer's ask and it is deliberately NOT scoped here, because it
  needs its own row, a `deepseek41` runtime to test against, and a decision
  about whether this project produces quantized artifacts at all.
- **Risk: the fit is tighter than 98.59 GiB suggests.** Engram is **39.8%** of
  the checkpoint, and **the two EMBEDDING TABLES — and only they — break the
  scale tiling.** `layers.{1,14}.engram.embed.scale` is `F8_E8M0 [rows, 8]`,
  which groups 32 columns and does NOT tile the rows, against the
  `[rows/32, cols/32]` tiling the `[32, 32]` linear weights use. The `wkv`
  scales in the same two shards do NOT break it: `engram.wkv.weight` is
  `F8_E4M3 [25600, 6144]` and `engram.wkv.scale` is `F8_E8M0 [800, 192]` =
  `[25600/32, 6144/32]` exactly. An earlier revision of this bullet said the
  scales of "engram" were laid out `[rows, 8]` without that restriction, which
  over-read two tensors into four. A residency plan that assumes uniform tiling
  will be wrong about the largest single component and right about the rest of
  engram.
  **That figure is measured, not estimated, and it replaces the "~36%" an
  earlier revision of this spec carried unsourced.** All twelve `engram` tensors
  sit in shards 47 and 48 (`model.safetensors.index.json`), and an HTTP range
  read of those two safetensors headers on 2026-09-11 gives
  `layers.{1,14}.engram.embed.weight` as `F8_E4M3 [384006168, 256]` and
  `[384016682, 256]` = 98,305,579,008 + 98,308,270,592 bytes, their
  `embed.scale` as `F8_E8M0 [rows, 8]` = 3,072,049,344 + 3,072,133,456, plus the
  two `wkv` pairs and four BF16 `[4, 5120]` projections: **203,073,076,240
  bytes = 189.127 GiB** against the index's own `total_size` of 510,286,023,000
  bytes (475.24 GiB) = **39.80%**. The producer measures the same component at
  "roughly 195 GiB of 473 GiB, about 41%" on ggml-org/llama.cpp#28696, which is
  his Q8_0 staging file rather than the release, so the two figures are
  consistent and neither is 36%.
- **Risk: the GGUF carries fields no config names.** `exp_probs_b_vl` (a second,
  vision-lane router bias) and per-layer `attn_sinks` appear in the file with no
  corresponding `config.json` key; their semantics are UNVERIFIED. A name map
  written from the config alone will not account for them.
- **Risk: the GGUF's pad token disagrees with the release.** The header says
  `tokenizer.ggml.padding_token_id = 1`; `config.json` says `pad_token_id = 2`.
  Unresolved, and load-bearing for any generation comparison.
- **Risk: the GGUF ships a chat template the HF repo does not.** Its origin is
  UNVERIFIED, so a prompt built from it is not a prompt the release defines.
- **Risk: the producer's card OMITS two complete rungs.** The card's "Files"
  table calls the ladder "Q2_K, Q3_K_M, Q4_K_M" and lists no other rung, while
  the tree at `543d86fd97` holds **five** complete rungs. Re-measured
  2026-09-11 from the HF tree API
  (`/api/models/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/543d86fd975044332d4366204eadd6da1b0cbeda?recursive=1&expand=1`,
  LFS sizes, paged by the `Link` header): Q1_0 3 shards 98.591 GiB, Q2_K 7
  shards 246.349 GiB, Q3_K_M 9 shards 323.422 GiB, Q4_K_M **11 shards 414.194
  GiB**, Q8_0 10 shards 473.069 GiB. The contrast is ONLY what the card omits —
  Q1_0 and Q8_0 appear nowhere in it. An earlier revision of this bullet set
  Q1_0 and Q8_0 against "Q4_K_M pending"; the card does still say `pending` in
  its Bytes column, but Q4_K_M is fully uploaded too, so `pending` is a stale
  card cell and not a third state. Read the tree, not the card.
- **Risk: the EXL3 checkpoint is partly the release's own bytes.** Verified by
  LFS oid: shards 1, 2 and 43-48 are byte-identical to `deepseek-ai`'s release
  (199.88 GiB); only 3-42 differ. EXL3 covers the routed experts only; fp8
  dense and attention, MXFP4 shared experts, the DSpark drafter, the fp8 engram
  tables and the native fp4 KV are the release's. "EXL3 3.5bpw" names an
  average over part of the file, not a format for the file.
- **Decision: "Pollard" is a recipe, not a format.** Hessian-aware sensitivity
  allocation over expert bit-widths; the bytes are turboderp EXL3 trellis. The
  exllamav3 revision that produced them is UNVERIFIED — the card names the
  `cuda-exl3 1.0.3` plugin and no upstream commit.

## Owed

- `QUANT-GGUF-Q1_0` stays `INVENTORIED`: the type is recognised and no
  executable path consumes it. Named here so the id-41 evidence has an owner.
- **W3b's MXFP8 host reference lands UNREACHED**, and this bullet is the
  AGENTS.md §"Nothing lands dead" record for it. `deepseek_v4_1_mxfp8.{h,cpp}`
  carries the five pieces of the MXFP8 32x32 UE8M0 linear family — the
  checkpoint-to-runtime scale row expansion, the runtime per-32-column dequant,
  the dynamic activation quantizer, the emulation linear arm (bf16 in and bf16
  out, `Mxfp8LinearEmulationBf16`, with an f32 reference beside it) and the
  `weight_scale` / `weight_scale_inv` name split. Nothing calls any of them:
  **W1** registered `deepseek_v41`, but its forward and its loader refuse by
  name, so no production entry point reaches this file. The wiring is owned
  by this row, split across **W8** (the loader arm, which calls the expansion and
  the name split) and **W4** (the host forward assembly, which calls the dequant
  and the linear arm). The W3b issue under this row's directory in
  [`.agents/issues/`](../issues/MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm/)
  tracks it and stays OPEN until a production entry point reaches the family; the
  landing commit names its ID. **It is deliberately not spelled here**, because
  `## Owed` carries ROWLESS issue IDs and `issue_records.py:518` refuses a
  row-owned ID in this section. AGENTS.md §"Nothing lands dead" and that checker
  are compatible in exactly one way: this section names the unreached SLICE, and
  the commit and pull-request bodies name the issue.
- **Two W3b guarantees are not gateable on a host, and both were measured, not
  assumed.** Mirroring upstream's MULTIPLY by `exp2(127 - sb)` rather than a
  divide by `exp2(sb - 127)` survives mutation with the binary proved changed,
  because the two forms are exact powers of two and only CDNA's flush-to-zero
  separates them (`mxfp8_utils.py:129-134`). The `amax` TINY floor survives for
  the same kind of reason: the low clamp already maps `log2(0) == -inf` to
  `sb == 0`. **W5's CUDA/ROCm arm owes both measurements**; until then the two
  are source-level fidelity decisions and this spec does not claim a gate holds
  them. A third fact belongs beside them: the upper clamp `254` is unreachable
  from any finite f32 (`ceil(log2(FLT_MAX / 448)) + 127 == 247`), so it bounds
  only a non-finite `amax`.
- The stale killgate-fork provenance in the reader, which lives at **three**
  comment sites and not one:
  [`gguf_reader.cpp:194`](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L194),
  [`:379`](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L379) and
  [`:388`](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L388) —
  that is `:194` plus `:378-392`, the span the `QUANT-GGUF-Q1_0` issue names.
  `:379` and `:388` each open "Killgate fork extension:", for `block_nvfp4`
  (id 40) and `block_q1_0` (id 41), and both are false at the same pin that
  falsifies the header. **Read the header comment precisely, because this bullet
  has now over-stated it once.** It sources the TRAITS for ids 39-41 from the
  killgate fork ("Ids 39-41
  follow mudler's killgate llama.cpp fork ... Block geometry from
  ggml/src/ggml-common.h"), but it attributes only **40 and 41** to that fork,
  as things it "appends ... after mainline's GGML_TYPE_MXFP4 = 39", and so it
  names **39 as mainline's** in its own words. What is stale is therefore the
  fork attribution of **40 and 41**, and only that: both are mainline at our
  `llama-cpp` pin `b10451`, re-derived 2026-09-12 by
  `git show b10451:ggml/include/ggml.h`, which prints `GGML_TYPE_MXFP4 = 39`
  (`:429`), `GGML_TYPE_NVFP4 = 40` (`:430`), `GGML_TYPE_Q1_0 = 41` (`:431`) and
  `GGML_TYPE_Q2_0 = 42` (`:432`). An earlier revision of this bullet said **two
  ids**, and that reading was the accurate one; do not re-widen the ID count to
  three. The SITE count is three, which is a different number. The owed work is
  therefore to rewrite ALL THREE comments, and to handle the
  unhandled mainline
  `GGML_TYPE_Q2_0 = 42` beside them. Product code, so it takes its own issue and
  its own change rather than riding a records commit. It is owned by the
  `QUANT-GGUF-Q1_0` row, whose issue is named in this spec's header, and it is
  deliberately NOT listed as an owed local ID here: `## Owed` carries rowless
  issues, and this one has a row.
- Every V4.1 capability above the registry rows — all **six** of the subsystems
  `## Upstream chain` declares and `## Our baseline` repeats: engram, MXFP8
  32x32, the indexer arm, DSpark, the vision tower, and the CED encoder-decoder
  KV split with its SWA Bounded Replay. The last one was missing from this list
  while `## Our baseline` was already calling it out as the polarity a later port
  must not discover late.
- **W3a's Engram host reference lands UNREACHED**, and this bullet is the record
  AGENTS.md §"Nothing lands dead" requires. `src/vllm/model_executor/models/`
  `deepseek_v4_1_engram.cpp` and its header are reached today only by
  `tests/vllm/models/test_deepseek_v4_1_engram.cpp`. **W1** registered
  `deepseek_v41`, but its forward and its loader refuse by name, so no production
  entry point can reach it: the loader arm is owed to **W8** and the
  `ModelRegistry::Forward` arm to **W4**.
  The wiring is owned by this row, whose W3a issue is named in the commit that
  landed the reference; it is deliberately NOT written as an owed local ID
  here, for the reason the `QUANT-GGUF-Q1_0` bullet above states — `## Owed`
  carries ROWLESS issues, and this one has a row.
- **The compressed token map CONSTRUCTION**, and it is owed rather than
  half-implemented. `build_compressed_token_map` (`common/engram.py:96-144`)
  needs `tokenizers`' NFKC, NFD, StripAccents and Lowercase normalizers, none of
  which this tree has, plus the `` (U+E000) private-use sentinel at `:109` that
  keeps a one-space token alive through `Strip()`. W3a therefore READS the map
  the published GGUF artifact carries as metadata
  (`deepseek41.engram.token_map`, 129,280 entries) and validates it. A
  partly-correct Unicode chain yields a plausible map that silently rehashes
  both tables, and upstream's own `build_compressed_token_map` is UNGATED — no
  test at `e77daef89e` touches it — so nothing ported would catch it. The
  sentinel is documented at our call site and is NOT gated by W3a.
- **The hash multiplier DERIVATION**, for the same reason. Upstream draws them
  from NumPy PCG64 seeded `10007 * layer_id`
  (`common/engram.py:147-166`); the artifact carries them
  (`multipliers`, 8 x u64), so W3a reads and validates instead of
  reimplementing one PRNG's bounded-integer path. The BOUND is ported, because
  the bound is what keeps `value * multiplier` inside int64. The upstream test
  parameter `compute_hash_multipliers((1, 14), 4, 99092)` is preserved by baking
  the eight values the oracle's own generator produces, with the recipe beside
  them.
- **An upstream defect W3a mirrors rather than fixes**:
  `common/engram.py:63-85` reports **61 composite**, because the witness 61 is
  a multiple of it. It is unreachable at the only call site, which starts its
  search at `engram_vocab_size - 1` = 15,999,999. Our port mirrors it and pins
  it in the gate. Revisit it only if a config ever lowers `engram_vocab_size`
  below 62.
- **`EngramLayout`'s prime loop and `build_compressed_token_map` are UNGATED
  upstream**, and there is no v4_1 end-to-end model test at `e77daef89e` at all.
  W3a's layout gate has no upstream counterpart to preserve, so it is ours and
  it is the only thing holding those 48 primes.
- **One upstream test case is DROPPED by W3a, and this bullet is its record**:
  `tests/kernels/test_engram.py::test_v2_model_state_gathers_lookback_window`
  (`:479-517` at `e77daef89e`). It exercises
  `DeepseekV41ModelState.prepare_inputs`, the runner-side gather that BUILDS
  `lookback_token_ids` from `all_token_ids` and `num_computed_tokens`, and W3a
  lands no model-state surface for it to drive. What it gates is the
  **newest-at-column-0** ordering of that window (`:509-514` expects
  `[22, 21, 16]` for a request whose history ends `..., 16, 21, 22`, and
  `:516-517` pins that graph-capture dummies share the same buffer). That
  ordering is exactly the contract `NgramHashState::Forward`'s tier 2 consumes
  at
  [`deepseek_v4_1_engram.cpp:417-425`](../../src/vllm/model_executor/models/deepseek_v4_1_engram.cpp#L417)
  (`col = chunk_start - 1 - lookback`), so W3a ASSUMES the producer's ordering
  on its own fixtures and proves nothing about the producer. Owed to **W4**,
  which is where the model state and its `prepare_inputs` land. Owned by this
  row, so it is deliberately NOT written as an owed local ID here, for the
  reason the `QUANT-GGUF-Q1_0` bullet above states.
- **W3c/W3d landed UNREACHED, and this bullet is the disclosure AGENTS.md
  §"Nothing lands dead" requires.** `deepseek_v4_1_indexer.{h,cpp}` carries the
  indexer-K host op, the q/kv RMSNorm, the layer and model topology and the
  per-layer RoPE selection. Its issue is row-owned and therefore NOT cited by id
  here -- `## Owed` carries rowless ids only, which `issue_records.py:519`
  enforces -- so the id lives in the landing commit body and in the file header,
  which is where AGENTS.md puts it.
  No production entry point calls any of them at their merge commit. The ROW
  that owes the wiring is this one,
  `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`; inside it, WAVE **W4**
  owns the host forward that composes W3a-W3f and is what will call them. W4 is
  a wave and not a row, and AGENTS.md §"Nothing lands dead" asks for the row, so
  both are written here.
  The gate is a unit gate over hand-built inputs, which proves that the
  functions work and never that anything reaches them.
- **Owed PERF, not a correctness gap: the MXFP8 query-quant fusion.**
  `query_quant.py`'s `fused_q_kv_rmsnorm_quant` folds the MXFP8 activation
  quantization into the norm, and `can_fuse_query_quant` (`query_quant.py:127`)
  refuses it off CUDA and again off a FlashInfer CUTLASS/CuTeDSL MXFP8 linear.
  `attention.py:602` therefore takes the plain `fused_q_kv_rmsnorm` branch
  everywhere a host reference runs, so the host port is CORRECT and the fusion
  is a Blackwell throughput arm that `W5` owns.
- **Owed record repair: W3d's cache counts.** The `## Work breakdown` W3d row
  says "Only FOUR compressed-KV caches and FOUR indexer-K caches exist (layers
  2/8/14/20)". The cache counts are right; what the row omits is that
  `index_source_layer_ids` has **EIGHT** entries `[2,8,14,20,24,28,32,36]`, so
  EIGHT layers carry an indexer while only four ALLOCATE a K cache. The other
  four (24/28/32/36) read layer 20's, because only a kv source owns the
  `wk`/`k_norm` pair that produces keys (`attention.py:371-395`). Measured
  2026-09-18 against the checked-in `config.json` fixture. The correction lives
  in `deepseek_v4_1_indexer.h` and is executable in
  `test_deepseek_v4_1_indexer.cpp`; rewriting the row's prose is owed to the
  next change that edits that table.

## Stop conditions

- Stop if a pin advance is proposed as part of this row. It is not in scope and
  it is not this row's decision.
- Stop before downloading any of the three artifacts. The smallest is 98.59 GiB
  and nothing here can consume it yet.
- Stop if a gate result is asked for. There is no oracle at this pin, and a
  number produced without one would be a measurement of nothing.
