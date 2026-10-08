# SPIKE: Gemma family (Gemma 1 / 2 / 3 / 4)

**STATUS: Gemma-1/2/3 text LANDED (SACRED gate 48/48 vs vLLM 0.25.0 each), rows
`ACTIVE` (correctness-complete, speed-pending); Gemma-4 is `BLOCKED` (W6 honesty
pass). The design below is the original spike, preserved as reference.** Grounded
the user's next model target after GLM ("and then we do gemma", said as "gemma 4").

**Base:** `d85fd04` (`Glm4MoeLiteForCausalLM` G1 landed). **Oracle pin:**
`/home/mudler/_git/vllm` @ `e24d1b24`. **dgx oracle:** `~/venvs/vllm-oracle` = vLLM
**0.25.0**. **Claim:** `CLAIM-SWEEP-GEMMA`. **Parent plan:**
[`breadth-sweep-plan.md`](breadth-sweep-plan.md) §B.3 Tier-2 rank 7 (Gemma3), extended
here to the whole family. **Gold-standard spike shape mirrored:**
[`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md).

Rows covered (Gemma-1/2/3 since advanced `INVENTORIED` -> `SPIKE` -> `ACTIVE`, gated 48/48; Gemma-4 -> `BLOCKED`):
`MODEL-TEXT-gemma-gemma-for-causal-lm`,
`MODEL-TEXT-gemma2-gemma2-for-causal-lm`,
`MODEL-TEXT-gemma3-gemma3-for-causal-lm`,
`MODEL-TEXT-gemma4-gemma4-for-causal-lm`.

Explicitly LEFT `INVENTORIED` (out of scope, stated with reason in §Scope):
`MODEL-TEXT-gemma3n-gemma3n-for-causal-lm` (the MatFormer/AltUp on-device nano arch —
a different architecture, not a Gemma-core delta); the embedding rows
`MODEL-EMBED-gemma2-*`/`MODEL-EMBED-gemma3-*`; every multimodal row
(`MODEL-MM-gemma3-mm-*`, `MODEL-MM-gemma3n-mm-*`, `MODEL-MM-gemma4-mm-*`,
`MODEL-MM-gemma4-unified-*`); and the MTP row `MODEL-SPEC-gemma4-mtp-*`.

---

## 0. Headline findings

### 0.0 The newest registered Gemma is **Gemma 4** — and it is real, with public checkpoints

The pin registers, in `vllm/model_executor/models/registry.py`, FIVE Gemma text
architectures and a full multimodal/MTP fan-out:
`GemmaForCausalLM` (`:105`), `Gemma2ForCausalLM` (`:106`), `Gemma3ForCausalLM`
(`:107`), `Gemma3nForCausalLM` (`:109`), **`Gemma4ForCausalLM` (`:110`)** — plus
`Gemma2Model`/`Gemma3TextModel` embeddings (`:216-217`),
`Gemma3ForConditionalGeneration` (`:383`), `Gemma3nForConditionalGeneration` (`:384`),
`DiffusionGemmaForBlockDiffusion` (`:388`), **`Gemma4ForConditionalGeneration`
(`:392`)**, **`Gemma4UnifiedForConditionalGeneration` (`:393`)**,
`PaliGemmaForConditionalGeneration` (`:513`), and `Gemma4MTPModel` (`:614`).

**Yes, a Gemma 4 exists in the pin, and it leads.** Public checkpoints exist (HF API,
fetched 2026-07-24, metadata only, nothing downloaded): `google/gemma-4-12B-it`
(11.96B, `Gemma4UnifiedForConditionalGeneration`, ungated), `google/gemma-4-31B-it`
(32.68B, `Gemma4ForConditionalGeneration`), `google/gemma-4-26B-A4B-it` (26.5B MoE,
`Gemma4ForConditionalGeneration`), plus community GGUF quants
(`unsloth/gemma-4-*-GGUF`). **But every public Gemma-4 checkpoint is
multimodal-wrapped** (`Gemma4*ForConditionalGeneration`/`Gemma4Unified…`); the bare
text row `Gemma4ForCausalLM` (`:110`) has **no standalone checkpoint** — to gate it
you extract the language-model backbone via the nested `text_config`
(`gemma4.py::_get_text_config`, `:206-215`).

### 0.1 What is genuinely new in Gemma-4 (why it leads the characterization but is NOT the first gate vehicle)

`gemma4.py` is **1715 lines** and is a different animal from the Gemma-1/2/3 dense
decoder. It composes, all in one text backbone:

- **Per-Layer Embeddings (PLE)** — a second embedding table `embed_tokens_per_layer`
  (`gemma4.py:986-991`) of width `hidden_size_per_layer_input * num_layers`, a
  `per_layer_model_projection` (`:1002-1010`), per-layer input gate + projection +
  norm inside every decoder layer (`:667-691`), scaled by `sqrt(ple_dim)` and combined
  `(proj + ple) * rsqrt(2)` (`:859-885`). Same family as Gemma3n's PLE.
- **YOCO (You Only Cache Once)** — the stack is split into a `self_decoder`
  (layers `0..K-1`) and a `cross_decoder` (`K..N-1`) with a fast-prefill path
  (`:785-950`, `:1191-1274`), gated on `cache_config.kv_sharing_fast_prefill`.
- **KV-sharing layers** — the last `num_kv_shared_layers` reuse an earlier same-type
  layer's KV via `kv_sharing_target_layer_name` (`:457-483`, `:522-534`).
- **A parallel MoE block** — `Gemma4Router` (weight-less `RMSNorm` + `root_size`
  `hidden^-0.5` + a learned per-dim `scale` + fp32 `GateLinear`, `:251-298`) feeding
  `Gemma4MoE` (softmax over ALL experts -> top-k -> renormalize, `per_expert_scale`
  folded into routing weights, custom Triton/torch routing, `:301-365`), running
  PARALLEL to the dense MLP and summed (`:725-734`).
- **`k_eq_v` layers** (K loaded into both K and V slots, no `v_proj`, `:567-580`,
  `:511-513`), a weight-less **V-norm** (`:431`), **double-wide MLP** on KV-shared
  layers (`:603-608`), and a per-layer **`layer_scalar`** buffer applied to every
  layer's output (`:694`, `:752`).
- **Q/K/V norms with `scaling=1.0`** — Gemma-4 drops `query_pre_attn_scalar` and lets
  the learnable Q/K norms carry the scale (`:402-405`).

`Gemma4ForCausalLM` e2e is therefore a **follow-on campaign, not a sweep bring-up**,
and it is **gate-BLOCKED as a first vehicle** by three independent facts, each mirrored
from how the GLM spike treated `Glm4MoeForCausalLM`:
1. **Checkpoints are multimodal-wrapped only** — the text row has no native
   checkpoint; gating means loading `language_model.*` out of a ≥12B MM checkpoint.
2. **Oracle-support is UNVERIFIED and likely absent.** Our correctness oracle is dgx
   vLLM **0.25.0**; Gemma-4 is very new (present at pin `e24d1b24`, far ahead of
   0.25.0). If 0.25.0 cannot construct `Gemma4*`, there is **no SACRED oracle** and no
   gateable target. **W0 verifies this before any Gemma-4 work is scheduled.**
3. **A large NEW-primitive stack** (PLE, YOCO, KV-sharing, the Gemma-4 MoE router,
   k_eq_v, double-wide MLP, per-layer scalar) that no other row needs.

### 0.2 The recent-first vehicle that FITS, is a clean text bring-up, and is oracle-certain: **Gemma 3**

`Gemma3ForCausalLM` (`gemma3.py`, 449 lines) is a clean dense decoder with genuine
pure-text checkpoints that are tiny and certain to run on the 0.25.0 oracle:
`google/gemma-3-1b-it` (1.0B, `Gemma3ForCausalLM`) and `google/gemma-3-270m` (268M,
`Gemma3ForCausalLM`). This is the **primary gate vehicle** — exactly the pattern the
GLM spike used (newest-registered `Glm4MoeForCausalLM` HW-blocked; the gate vehicle
was the variant that fit).

### 0.3 Gemma reduces MOSTLY to existing infrastructure — the two primitives the task flagged are BOTH already ours, and more

Verified in `src/`/`include/` on base `d85fd04`:

- **gemma-RMSNorm (the `(1 + weight)` offset, variance in fp32)** is ALREADY an op:
  `vt::RmsNorm(..., {eps, gemma=true}, residual)` — `include/vt/recipes.h:22-42,55-73`
  applies the weight as `(1 + w)` for the GemmaRMSNorm subclass and is the SAME op the
  GLM-4 landing uses. **REUSE.** (This is the single most-used norm across all four
  Gemma versions: input/post-attention/pre-ff/post-ff, q/k/v norms, final norm.)
- **Sandwich norms** — the extra RMSNorm on a sublayer's OUTPUT before the residual
  add — LANDED for GLM-4-9B G2 (`b568d20`): `src/vllm/model_executor/models/glm4.cpp:155-188`
  is exactly `post_self_attn_layernorm`/`post_mlp_layernorm` as standalone
  `vt::RmsNorm` on the attn/mlp output. Gemma-2/3's `post_attention_layernorm`
  (STANDALONE), `pre_feedforward_layernorm`, `post_feedforward_layernorm`
  (`gemma3.py:254-289`) are the SAME pattern. **REUSE — a cross-family additivity
  proof, GLM -> Gemma.**
- **SentencePiece tokenizer** — `src/vllm/tokenizer/tokenizer.cpp:185-199,459-485`
  detects the `Metaspace` pre-tokenizer and selects `Family::kSentencePiece`, and its
  own comments name **"Mistral/Gemma"** as the family. BOS is extracted via
  `ExtractBosEos` (`:311-360`) and `PrependScheme` (`:463-467`). **REUSE** — with a W0
  guard: Gemma prepends BOS, and the OPT lesson (a silently-unapplied BOS scores 0/n
  while emitting fluent text) means BOS placement is verified against the oracle at W0,
  not assumed.
- **Sliding-window / interleaved local+global attention** — the FA-2 kernel already
  supports a finite local window: `src/vt/cuda/cuda_flash_attn_fa2.cu:397-407`
  (`is_local`, `window_size_left/right`, LOCAL_SWITCH), and the KV layer accepts it
  (`src/vllm/model_executor/layers/attention/attention.cpp:13-23`,
  `per_layer_sliding_window`/`model_sliding_window`). The KV-cache specs exist too:
  `SlidingWindowSpec` and `ChunkedLocalAttentionSpec`
  (`src/vllm/v1/kv_cache_interface.cpp:76-95`,
  `src/vllm/v1/kv_cache_spec_registry.cpp:35-38`). **REUSE at the kernel + KV-spec
  level.** The NEW part is only the *wiring*: per-layer routing by
  `config.layer_types[i]` and a second RoPE cache for the local layers (§0.5).
- **QK-norm + RoPE fused preamble** — `kAttnQkNormRopeGate`
  (`include/vt/recipes.h:231-244`) is literally "gemma-RMSNorm(q) + gemma-RMSNorm(k) +
  partial NeoX RoPE from a cos/sin cache", transcribed from vLLM's
  `QKNormRoPEFusionPass`, already hand-called by qwen3/qwen3_5. Gemma-3/4 QK-norm
  (`GemmaRMSNorm`/`RMSNorm` on `head_dim`, then NeoX rope) is this exact shape.
  **REUSE.**
- **Tied word embeddings** — `tie_word_embeddings` handling exists on opt/qwen3_dense/
  glm4 loaders (`grep` hits in `opt_weights.cpp`, `qwen3_dense.cpp`, `glm4.cpp`).
  Gemma ties embed <-> lm_head (all four versions; `gemma.py:339,350`,
  `gemma3.py:411-412`). **REUSE.**

### 0.4 What is genuinely NEW for the Gemma-3 gate vehicle (a small, additive set)

1. **GeGLU activation (`gelu_pytorch_tanh` + multiply).** We have ONLY SiLU-family
   activations — `include/vt/ops.h:91` `kSiluAndMul`, `:537` `SiluAndMulFn`; a grep for
   any `gelu`/`Gelu` in `src/`/`include/` returns **zero**. Gemma's MLP is `GeluAndMul`
   with the tanh approximation (`gemma3.py:87` `get_act_and_mul_fn(hidden_activation)`,
   `gemma.py:78-82`, `gemma2.py:86`). This is the ONE genuinely-new compute kernel:
   `kGeluAndMul` (tanh-approx GeLU on the gate half, elementwise multiply by the up
   half). Gemma-4's MoE also uses `activation="gelu_tanh"` (`gemma4.py:361`).
2. **Final logit soft-cap.** A grep for `soft_cap`/`softcap`/`tanh` in
   `include/vllm/v1/sample/logits_processor/builtin.{h,cpp}` and `sampler.cpp` returns
   **nothing** — we do not cap output logits. Gemma-2/3 pass
   `soft_cap=config.final_logit_softcapping` to the logits processor
   (`gemma2.py:344-345`, `gemma3.py:414-416`): `logits = cap * tanh(logits / cap)`.
   NEW, small, in the logits/sampler path. (Gemma-2 uses `30.0`; Gemma-3 configs
   usually set it `null`, so this primitive is *proved* by the Gemma-2 gate, §0.6.)
3. **`query_pre_attn_scalar` attention scaling + `sqrt(hidden)` embedding normalizer.**
   Gemma-2/3 scale attention by `query_pre_attn_scalar**-0.5` (not `head_dim**-0.5`)
   (`gemma2.py:129`, `gemma3.py:130`) and multiply the token embeddings by
   `sqrt(hidden_size)` cast to bf16 (`gemma3.py:328-341`). Both are scalars — NEW but
   trivial. (Gemma-4 sets `scaling=1.0` and lets the Q/K norms carry it, `:402-405`.)
4. **Dual per-layer RoPE theta + interleaved sliding-window routing (wiring).**
   Gemma-3 uses `rope_theta` on full-attention layers and `rope_local_base_freq` on
   sliding layers (`gemma3.py:158-176`), selected by `config.layer_types[i]`. We have
   per-layer RoPE-from-cache already (`RopeFromCache`, used for GLM decoupled rope);
   the NEW part is building TWO cos/sin caches (global + local theta) and routing per
   layer, plus routing the sliding window per layer. No new kernel — a second cache +
   a per-layer selector.

### 0.5 The per-Gemma-version delta (get attn soft-cap vs QK-norm right)

| Version | Attn logit soft-cap | Final logit soft-cap | Q/K-norm | Attn scaling | Sandwich norms | Sliding window | Activation | New for us |
|---|---|---|---|---|---|---|---|---|
| **Gemma 1** (`gemma.py`) | none | none (plain `LogitsProcessor`) | none | `head_dim**-0.5` | only `post_attention_layernorm` (fused add) | none | GeGLU tanh | GeGLU + embed-scale |
| **Gemma 2** (`gemma2.py`) | **yes** (`attn_logit_softcapping`, `:202`) | **yes** (`final_logit_softcapping`, `:345`) | none | `query_pre_attn_scalar**-0.5` | **yes** (pre/post-ff standalone, `:217-245`) | **yes** (interleaved, `:154-166`) | GeGLU tanh | + attn soft-cap wiring + final soft-cap + qpas |
| **Gemma 3** (`gemma3.py`) | **removed** (`attn_logits_soft_cap=None`, `:243`) | supported but usually `null` | **yes** (`GemmaRMSNorm` on head_dim, `:149-150,211-216`) | `query_pre_attn_scalar**-0.5` | yes | yes (dual rope theta, `:158-176`) | GeGLU tanh | + QK-norm (reuse recipe) + dual rope |
| **Gemma 4** (`gemma4.py`) | optional (`attn_logit_softcapping`, `:592`) | via logits proc | **yes** (Q/K + weight-less V-norm, `:428-431`) | **`1.0`** (norms carry it, `:405`) | yes (+ MoE post-ff-1/2) | yes | GeGLU tanh + MoE gelu_tanh | PLE + YOCO + Gemma-4 MoE + k_eq_v (campaign) |

**The load-bearing per-version fact:** Gemma-2 has an ATTENTION logit soft-cap;
Gemma-3 REMOVED it and added QK-norm instead. So the Gemma-3 gate vehicle needs
**QK-norm (reuse `kAttnQkNormRopeGate`)** and does NOT need the attention soft-cap
wiring; the attention soft-cap is proved separately by the Gemma-2 gate. The FA-2
kernel already carries a soft-cap capability (`cuda_flash_attn_fa2.cu` is in the
soft_cap file set), but our `Attention` layer ctor does not currently thread
`logits_soft_cap` through — that wiring is the Gemma-2 delta, not the Gemma-3 one.

### 0.6 Hardware fit — measured, per variant

Sizes from the HF API 2026-07-24 (metadata only; **nothing downloaded**). GB10 budget:
~119 GiB unified memory (task notes ~42 GiB practical run headroom after G1). dgx free
disk: 184 GiB at 95% (per the GLM spike's measured correction; staged downloads only).
bf16 on-disk ≈ 2 x params. **All `google/gemma-*` repos are gated (manual HF
approval)** — an operational W0 note; the ungated `google/gemma-4-*` repos are not.

| Variant | Smallest genuine checkpoint | Params | bf16 on disk | Arch on the checkpoint | GB10 verdict |
|---|---|---|---|---|---|
| `Gemma3ForCausalLM` (dense) | **`google/gemma-3-270m`** | 268M | ~0.5 GiB | `Gemma3ForCausalLM` (pure text) | **FITS trivially — tiny secondary/CI-scale vehicle** |
| `Gemma3ForCausalLM` (dense) | **`google/gemma-3-1b-it`** | 1.00B | ~2.0 GiB | `Gemma3ForCausalLM` (pure text) | **FITS — recommended PRIMARY gate vehicle** |
| `Gemma3ForCausalLM` (bigger strict) | `google/gemma-3-4b-it` | 4.30B | ~8.6 GiB | `Gemma3ForConditionalGeneration` (text backbone `Gemma3`) | **FITS — bigger-model strict check (text backbone)** |
| `Gemma2ForCausalLM` (attn+final soft-cap) | **`google/gemma-2-2b-it`** | 2.61B | ~5.2 GiB | `Gemma2ForCausalLM` (pure text) | **FITS — proves attn+final soft-cap + qpas** |
| `Gemma2ForCausalLM` (bigger strict) | `google/gemma-2-9b-it` | 9.24B | ~18.5 GiB | `Gemma2ForCausalLM` (pure text) | **FITS — Gemma-2 strict check** |
| `GemmaForCausalLM` (Gemma 1) | `google/gemma-1.1-2b-it` | 2.51B | ~5.0 GiB | `GemmaForCausalLM` (pure text) | **FITS — lowest priority** |
| `Gemma4ForCausalLM` (PLE+YOCO+MoE, text backbone) | `google/gemma-4-12B-it` | 11.96B | ~23.9 GiB | `Gemma4UnifiedForConditionalGeneration` (MM) | **fits memory, but gate-BLOCKED as first vehicle** — MM-wrapped only, oracle-support UNVERIFIED on 0.25.0, PLE/YOCO/MoE campaign |
| `Gemma4ForCausalLM` (dense 31B) | `google/gemma-4-31B-it` | 32.68B | ~65 GiB | `Gemma4ForConditionalGeneration` (MM) | **HW-MARGINAL/BLOCKED** vs the ~42 GiB run headroom; MM + campaign |
| `Gemma4` MoE (26B-A4B) | `google/gemma-4-26B-A4B-it` | 26.54B | ~53 GiB | `Gemma4ForConditionalGeneration` (MM MoE) | **HW-BLOCKED** vs run headroom; MM + campaign |
| `Gemma3nForCausalLM` (MatFormer) | `google/gemma-3n-E2B-it` | 5.44B | ~10.9 GiB | `Gemma3nForConditionalGeneration` (MM) | out of scope (separate arch, stays `INVENTORIED`) |

**No proposed gate depends on hardware we do not have.** The primary vehicle
(`gemma-3-1b-it`, ~2 GiB) fits any budget; `gemma-3-270m` is small enough for the CPU
CI tier. Presence is NOT read from a table (the OPT lesson): W0 verifies weight FILES
before any row is picked up.

### 0.7 Recommended order

```
  W0  Ground facts on HW: verify 0.25.0 oracle constructs Gemma3 (and probe Gemma4);
      fetch gemma-3-1b-it; confirm live config; run oracle + K=5 greedy self-determinism
      (selects gate form); verify BOS placement vs oracle. No code.
        |
  W1  GeGLU activation (kGeluAndMul, tanh) + sqrt(hidden) embed-scale + query_pre_attn_scalar.
      Additive (no existing model sets them -> all 10 regressions byte-identical). Unit-gated.
        |
  W2  Gemma3 dense block (gemma3.{h,cpp}): sandwich norms (REUSE glm4), QK-norm (REUSE
      kAttnQkNormRopeGate), dual per-layer rope theta + interleaved sliding-window routing
      (REUSE FA2 window + SlidingWindow/ChunkedLocalAttention specs), tied embeddings.
      -> SACRED gate on gemma-3-1b-it (form per W0) + bigger strict on gemma-3-4b text backbone.
        |
  W3  Final logit soft-cap in the logits path (+ Gemma2 attn soft-cap wiring through Attention).
        |
  W4  Gemma2ForCausalLM (gemma2.{h,cpp}): attn+final soft-cap + qpas.
      -> SACRED on gemma-2-2b-it, STRICT on gemma-2-9b-it. Proves the soft-cap primitives.
        |
  W5  GemmaForCausalLM (Gemma 1): simplest (no sandwich beyond post-attn, no soft-cap, no QK-norm).
      -> SACRED on gemma-1.1-2b-it. Lowest priority.
        |
  W6  Gemma4 characterization/honesty pass: oracle-support verdict on 0.25.0; config/registry
      resolution from the real MM text_config; the PLE/YOCO/MoE/k_eq_v primitive inventory as a
      scoped follow-on campaign. Record HW/complexity honestly. NO implementation unless the
      oracle can construct it AND a text-backbone checkpoint fits.
        |
  W7  Speed close: decode-graph sibling per landed Gemma + the every-axis grid vs vLLM.
      Nothing reaches DONE before this.
```

G1 = W2 (Gemma-3) is the highest-value item and the first token-exact gate. W1 is
fully independent (pure additive ops) and unblocks all four versions; sequence it
first. W6 is a characterization pass, not implementation, until its two preconditions
(oracle support + a fitting text-backbone checkpoint) are both met.

---

## 1. Structured contract

### Scope

Design — not build — the Gemma text family end to end (Gemma 1/2/3/4), determine the
recent-first gate vehicle that actually runs on GB10 against the pinned oracle, and
factor the work honestly into what REUSES landed infrastructure versus what is
genuinely new. Covers `MODEL-TEXT-gemma-gemma-for-causal-lm`,
`MODEL-TEXT-gemma2-gemma2-for-causal-lm`, `MODEL-TEXT-gemma3-gemma3-for-causal-lm`,
`MODEL-TEXT-gemma4-gemma4-for-causal-lm`.

In scope: the complete Gemma registry inventory with per-version characterization; the
reuse-vs-new factoring (gemma-RMSNorm, sandwich norms, SentencePiece, sliding-window,
QK-norm, tied embeddings all REUSE; GeGLU, final logit soft-cap, qpas + embed-scale,
dual rope routing all NEW); the per-version soft-cap-vs-QK-norm delta; measured
per-variant GB10 hardware fit; the Gemma-4 primitive stack (PLE, YOCO, KV-sharing, the
Gemma-4 MoE router, k_eq_v, double-wide MLP, layer scalar) with its gate-BLOCKED
verdict; the SACRED gate design (form BY MEASUREMENT); and the upstream test inventory.

OUT of scope, each with a reason: **implementation of anything** (this is a spike — no
code, no kernels, no build, no benchmark, nothing downloaded). **`Gemma4ForCausalLM`
e2e**, because its only checkpoints are multimodal-wrapped (≥12B), its 0.25.0
oracle-support is unverified, and it needs a PLE/YOCO/MoE/k_eq_v primitive stack no
other row uses — it is a scoped follow-on campaign, not a first gate vehicle.
**`Gemma3nForCausalLM`**, because the MatFormer/AltUp/Laurel nano architecture is a
distinct design, not a Gemma-core delta; it stays `INVENTORIED`. **All Gemma multimodal
rows** (`gemma3_mm`, `gemma3n_mm`, `gemma4_mm`, `gemma4_unified`, `paligemma`,
`diffusion_gemma`), because the vision/audio tower tracks have not started; they stay
`INVENTORIED`. **The Gemma-4 MTP row** (`Gemma4MTPModel`), because speculative decoding
is a separate track. **The Gemma embedding rows**, because pooling/embedding is a
separate modality.

### Upstream chain

Registry (`vllm/model_executor/models/registry.py` @ `e24d1b24`): `:105`
`GemmaForCausalLM` (`gemma.py`); `:106` `Gemma2ForCausalLM` (`gemma2.py`); `:107`
`Gemma3ForCausalLM` (`gemma3.py`); `:109` `Gemma3nForCausalLM` (`gemma3n.py`); `:110`
`Gemma4ForCausalLM` (`gemma4.py`); embeddings `:216-217`; multimodal `:383` (gemma3_mm),
`:384` (gemma3n_mm), `:388` (diffusion_gemma), `:392` (gemma4_mm), `:393`
(gemma4_unified), `:513` (paligemma); MTP `:614` (gemma4_mtp).

Model layer: `gemma.py:57-388` (`GemmaMLP`/`GemmaAttention`/`GemmaDecoderLayer`/
`GemmaModel`/`GemmaForCausalLM`; act `:59-86`, embed-scale `:288-295`, tie `:339,350,386`);
`gemma2.py:57-378` (attn soft-cap `:106,165,202`, qpas `:129`, sandwich `:213-245`,
sliding `:154-166`, final soft-cap `:344-345`, embed-scale `:276-283`, tie `:338-339,376`);
`gemma3.py:63-449` (QK-norm `:149-150,211-216`, no attn soft-cap `:243`, dual rope
`:158-176`, sliding `:152-155`, sandwich `:254-289`, qpas `:130`, embed-scale
`:328-341`, final soft-cap `:414-416`, tie `:411-412,447`);
`gemma4.py:88-1418` (routing kernel `:92-203`, `Gemma4Router` `:251-298`, `Gemma4MoE`
`:301-365`, `Gemma4Attention` incl. k_eq_v/KV-share/V-norm/scaling=1 `:368-539`,
`Gemma4DecoderLayer` incl. MoE-parallel/PLE/layer_scalar `:542-754`, YOCO
`:757-950,1063-1099`, PLE model wiring `:965-1037`, forward `:1276-1357`, loader
`:1359-1418+`).

Norm: `vllm/model_executor/layers/layernorm.py::GemmaRMSNorm` (the `(1+w)` fp32
subclass). Activation: `vllm/model_executor/layers/activation.py::GeluAndMul(approximate="tanh")`
and `get_act_and_mul_fn`. RoPE: `vllm/model_executor/layers/rotary_embedding/get_rope`.
Attention + sliding window + soft-cap: `vllm/model_executor/layers/attention/Attention`
(`logits_soft_cap`, `per_layer_sliding_window`, `kv_sharing_target_layer_name`).
Logits soft-cap: `vllm/model_executor/layers/logits_processor.py::LogitsProcessor(soft_cap=...)`.

### Our baseline

REUSED as-is (each with the anchor and why):
- **gemma-RMSNorm `(1+w)` fp32** — `include/vt/recipes.h:22-42,55-73`; the same
  `vt::RmsNorm {gemma=true}` op GLM-4 uses. Covers every Gemma norm.
- **Sandwich norms** — `src/vllm/model_executor/models/glm4.cpp:155-188` (standalone
  `vt::RmsNorm` on sublayer output before the residual add), landed `b568d20`. Gemma-2/3
  pre/post-ff norms are the same pattern -> cross-family reuse.
- **SentencePiece tokenizer + BOS** — `src/vllm/tokenizer/tokenizer.cpp:185-199,321-370,459-485`
  (Metaspace -> `kSentencePiece`, names "Mistral/Gemma"; `ExtractBosEos`;
  `PrependScheme`).
- **Sliding-window / local attention** — FA-2 window
  `src/vt/cuda/cuda_flash_attn_fa2.cu:397-407`; layer plumbing
  `src/vllm/model_executor/layers/attention/attention.cpp:13-23`; KV specs
  `SlidingWindowSpec`/`ChunkedLocalAttentionSpec`
  (`src/vllm/v1/kv_cache_interface.cpp:76-95`,
  `src/vllm/v1/kv_cache_spec_registry.cpp:35-38`).
- **QK-norm + rope fused preamble** — `include/vt/recipes.h:231-244` `kAttnQkNormRopeGate`
  (gemma-RMSNorm(q/k) + NeoX rope), hand-called by qwen3/qwen3_5.
- **Tied embeddings** — the `tie_word_embeddings` skip-`lm_head` path on
  opt/qwen3_dense/glm4 loaders.
- **Dense glue + GQA paged path** — `include/vllm/model_executor/models/dense_attn_block.h`
  glue (`Dev`/`DBuf`/`DevicePool`/`ResidentWeight`/`KvSlice`/`BuildStepInputs` + the
  GQA paged forward), the model-factory `REGISTER_VLLM_MODEL` seam, and the
  `ENG-RUNNER-MODELSHAPE` model-shape-agnostic runner.
- **The decode CUDA-graph driver pattern** (`Qwen3DecodeGraph`/`Qwen3_5DenseDecodeGraph`
  siblings over `decode_graph_sizes.h`) for the W7 speed close.
- **BF16 grouped-MoE GEMM** — `vt::MoeGroupedGemmBf16` + `RunMoeBlock`
  (`include/vllm/model_executor/models/qwen3_5_moe_block.h:45`) for the Gemma-4 MoE
  block IF that campaign lands (needs a Gemma-4-specific router: softmax-over-all +
  `per_expert_scale`, distinct from the sigmoid/noaux_tc GLM/DeepSeek router).

Honestly NOT reusable, and why:
- **GeGLU activation** — we have NONE (only SiLU-family `kSiluAndMul` at
  `include/vt/ops.h:91`; zero `gelu` hits). NEW kernel `kGeluAndMul` (tanh approx).
- **Final logit soft-cap** — no `soft_cap`/`tanh` in
  `include/vllm/v1/sample/logits_processor/builtin.{h,cpp}` or `sampler.cpp`. NEW,
  small, in the logits path.
- **Attention-layer soft-cap wiring** — the FA-2 kernel has the capability but our
  `Attention` ctor does not thread `logits_soft_cap` through. NEW wiring (Gemma-2/4
  only; Gemma-3 gate vehicle does not need it).
- **`dense_attn_block.h::AttnBlock` body** — it hard-codes Qwen-style qk-norm + NeoX
  rope and does not carry sandwich norms, dual rope theta, per-layer sliding routing,
  or the embed-scale/qpas scalars. Per the OPT + GLM precedent (D4), Gemma gets a NEW
  `gemma3.{h,cpp}` block that reuses only the glue and the GQA paged path — it does NOT
  extend `AttnBlock`.
- **The entire Gemma-4 primitive stack** (PLE, YOCO self/cross split, KV-sharing, the
  Gemma-4 MoE router, k_eq_v, double-wide MLP, per-layer scalar) — all NEW; a scoped
  follow-on campaign, not this bring-up.
- **No MODEL row is `DONE`** anywhere in the matrix; the best states are `ACTIVE`
  (correctness complete, speed pending). Nothing here claims otherwise.

Precedent specs: [`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md) (the
gold-standard spike shape + the sandwich-norm and cross-family additivity precedent),
[`sweep-qwen3-coder-30b.md`](sweep-qwen3-coder-30b.md) (BF16 MoE bring-up shape),
[`first-additive-model-qwen3-dense.md`](first-additive-model-qwen3-dense.md) (the dense
per-row protocol W0..W5), [`sweep-opt-125m.md`](sweep-opt-125m.md) (the BOS/tokenizer
lesson + cross-family additivity canary).

**Anchor-drift warning.** Line anchors move between bases. Re-anchor every cited
`file:line` against the tree at implementation time; `check_links` validates line
ranges. The `gemma*.py` anchors above are against pin `e24d1b24`.

### Port map

| Upstream | Ours |
|---|---|
| `registry.py:107` `Gemma3ForCausalLM` (`gemma3.py:63-449`) | **NEW** `include/vllm/model_executor/models/gemma3.h` + `src/vllm/model_executor/models/{gemma3_registry,gemma3_weights,gemma3}.cpp` — one `REGISTER_VLLM_MODEL`, full-attn+sliding KV spec, `is_dense_model=true`. **The gate vehicle.** |
| `registry.py:106` `Gemma2ForCausalLM` (`gemma2.py:57-378`) | **NEW** `gemma2.{h,cpp}` — the Gemma-3 block minus QK-norm, plus attn+final soft-cap and qpas. Reuses W1/W2/W3 pieces |
| `registry.py:105` `GemmaForCausalLM` (`gemma.py:57-388`) | **NEW** `gemma.{h,cpp}` — simplest: post-attn fused norm only, no soft-cap, no QK-norm, GeGLU + embed-scale. Lowest priority |
| `registry.py:110` `Gemma4ForCausalLM` (`gemma4.py`) | **NEW (campaign, gate-BLOCKED as first vehicle)** `gemma4.{h,cpp}` — PLE + YOCO + Gemma-4 MoE router + k_eq_v + double-wide MLP + layer scalar. Characterization/honesty pass only until oracle-support + a fitting text-backbone checkpoint are both proven (W6) |
| `layernorm.py::GemmaRMSNorm` | **REUSE** `vt::RmsNorm {gemma=true}` (`recipes.h:22-42`) |
| `gemma3.py:254-289` sandwich norms | **REUSE** the GLM-4 standalone-norm pattern (`glm4.cpp:155-188`) |
| `activation.py::GeluAndMul(approximate="tanh")` | **NEW** `kGeluAndMul` op (tanh-approx GeLU + mul) in `include/vt/ops.h` + `src/vt/cuda/` + `src/vt/cpu/`. **Shared primitive** (Gemma 1/2/3/4 + Gemma-4 MoE) |
| `gemma2.py:345` / `gemma3.py:415` final logit soft-cap | **NEW** `soft_cap` (tanh cap) in the logits/sampler path |
| `gemma3.py:130` qpas scaling + `:328-341` `sqrt(hidden)` embed-scale | **NEW** two scalars threaded into the new block (additive, no-op for existing models) |
| `gemma3.py:158-176` dual rope theta + `:152-155` per-layer sliding | **NEW wiring** — a second cos/sin cache (local theta) + per-layer routing over `layer_types`; REUSES `RopeFromCache` + the FA-2 window |
| `gemma3.py:149-150,211-216` QK-norm | **REUSE** `kAttnQkNormRopeGate` (`recipes.h:231-244`) |
| tied embed <-> lm_head | **REUSE** the existing `tie_word_embeddings` loader path |
| SentencePiece tokenizer + BOS | **REUSE** `tokenizer.cpp` `kSentencePiece` path; W0 verifies BOS vs oracle |
| `gemma4.py` PLE / YOCO / MoE router / k_eq_v / double-wide MLP / layer scalar | **NOT PORTED here** — scoped in §0.1 as the Gemma-4 follow-on campaign; W6 records the design + the oracle/HW verdict |
| `gemma3n.py`, `gemma*_mm.py`, `gemma4_mtp.py`, embeddings | **NOT PORTED** — out of scope; stay `INVENTORIED` |

### Tests to port

Per [`.agents/porting.md`](../porting.md). Unlike GLM (which had almost no
upstream text-correctness coverage), **Gemma HAS upstream text-generation correctness
entries** — a genuine executable spec. Nothing below is ported by this spike (spec
only); this is the inventory that binds the implementing Ws.

| Upstream test | Tier | Ours |
|---|---|---|
| `tests/models/language/generation/test_common.py:54` (`google/gemma-1.1-2b-it`) | T-parity | `tests/vllm/models/test_gemma_paged_engine.cpp` (W5) |
| `tests/models/language/generation/test_common.py:62` (`google/gemma-2-2b-it` — "test hybrid attention", i.e. interleaved sliding/global) | T-parity | `tests/vllm/models/test_gemma2_paged_engine.cpp` (W4) — the interleaved-attention spec |
| `tests/models/language/generation/test_common.py:173-178` (Gemma 1/2 prompt-normalization / BOS handling) | T-parity | the W0 BOS verification, encoded as a tokenizer-parity case |
| `tests/v1/e2e/general/test_correctness_sliding_window.py` | T-e2e | sliding-window correctness for the Gemma-3 interleaved layers (W2) |
| `tests/v1/e2e/general/test_kv_sharing_fast_prefill.py` | T-e2e | SKIPPED — Gemma-4 YOCO/KV-sharing (W6 campaign) |
| `tests/models/registry.py` `_HfExamplesInfo` for each Gemma arch | T-unit | config/registry resolution cases per Gemma row |
| `tests/models/test_initialization.py` (Gemma init smoke) | T-unit | init/registry resolution; the Gemma-4 case gates the 0.25.0 oracle-support verdict |
| `tests/test_config.py` (Gemma arch-config resolution) | T-unit | arch-config resolution, gateable with NO checkpoint |
| `tests/tool_parsers/test_gemma4_tool_parser.py`, `test_functiongemma_tool_parser.py` | T-unit | Serving-layer parsers; deferred to the tool-calling track, inventoried here |

**Upstream coverage note:** the Gemma-2 entry is explicitly labelled "test hybrid
attention" — the interleaved sliding/global pattern IS the upstream spec for the
sliding-window wiring, and it is directly portable. This is a stronger executable spec
than GLM had; our SACRED oracle is still our own pinned-vLLM comparison per
[`.agents/verification.md`](../verification.md), but the interleaved-attention behaviour is
upstream-gated.

### Gates

1. **Correctness (SACRED), `Gemma3ForCausalLM` on `google/gemma-3-1b-it` bf16.**
   Token-exact vs the pinned vLLM 0.25.0 oracle, greedy, identical prompt set. Gate
   form selected BY MEASUREMENT per [`near-tie-distributional-gate`](../verification.md): run
   vLLM's own greedy K=5 first. A 1.0B dense model is in the small-dense near-tie
   regime (cf. Qwen3-0.6B), so a distributional (our token in vLLM's K-run set) fallback
   is *permitted only if measured*; otherwise STRICT.
2. **Correctness, bigger-model STRICT.** `Gemma3ForCausalLM` on `google/gemma-3-4b-it`
   (text backbone), where vLLM 0.25.0 greedy is expected deterministic -> STRICT
   token-exact. This is the primitive-proof gate: a silently-unapplied rope slice,
   sandwich norm, embed-scale, or BOS emits fluent text while being wrong (the OPT
   failure mode), so an e2e token-exact gate is required, not a unit test alone.
3. **New ops, unit-gated at REAL dims against a CPU reference.** `kGeluAndMul`
   (tanh-approx GeLU + mul) at Gemma-3-1b intermediate size; final logit soft-cap
   (`cap * tanh(logits/cap)`) at `final_logit_softcapping=30.0`; `query_pre_attn_scalar`
   scaling and `sqrt(hidden)` embed-scale asserting bit-exactness; dual per-layer rope
   theta asserting the local-theta cache is applied on sliding layers and the global on
   full-attention layers; the non-roped/passthrough invariants preserved.
4. **Loader.** Weight-map coverage with zero unmapped and zero missing tensors for every
   gated checkpoint, including the tied `lm_head <- embed_tokens` skip.
5. **Regression, non-negotiable — all 10 current SACRED gates UNCHANGED.** 27B 235/235,
   35B 315/315, Qwen3-Coder 6/6, Qwen3-dense (0.6B+4B 16/16), OPT-125m 6/6,
   DeepSeek-V2-Lite 8/8, Llama-3.2-1B 16/16, Mistral-7B-v0.3 16/16, GLM-4-9B-0414 16/16,
   GLM-4.7-Flash 8/8. Every op added (GeGLU, soft-cap, scalars, dual rope) is additive
   and default-inert for existing models; a regression on any voids the change.
6. **Build.** Clean full rebuild `-Werror`, zero warnings. Per
   [`incremental-build-masks-werror`](../workflow.md), the gate is a clean rebuild
   (header changes are certain here), not an incremental one.
7. **memcheck.** `compute-sanitizer` zero errors on the new kernels (`kGeluAndMul`).
8. **Records.** `scripts/check-agent-record.py`, `scripts/check-doc-checkpoint.py`,
   `scripts/check-device-leakage.py`, `scripts/check-readme-structure.py`,
   `scripts/check-model-checklist.py` all green.
9. **SPEED.** Explicitly PENDING and unclaimed for every row. Per the acceptance rule,
   a model is `DONE` only at token-exact AND vLLM throughput on every axis; no row here
   reaches `DONE` on correctness alone.
10. **Blocked-row honesty gate (Gemma-4).** For `Gemma4ForCausalLM` the gateable subset
    is: the 0.25.0 oracle-support verdict; config/registry resolution from the real MM
    `text_config`; and the PLE/YOCO/MoE/k_eq_v primitive inventory as a scoped campaign.
    It records `SPIKE`/`BLOCKED` with the measured reasons and never claims more.

### Dependencies

**No hard upward dependency.** Unlike the GLM MLA vehicle, every Gemma-3/2/1 gate is a
pure dense-path bring-up on landed infrastructure (gemma-RMSNorm, sandwich norms,
SentencePiece, sliding window, QK-norm recipe, tied embeddings, the model-shape-agnostic
runner). The only genuinely-new compute kernel is `kGeluAndMul`.

**Checkpoint dependencies (downloads, not yet performed, all HF-gated for `google/*`):**
`gemma-3-1b-it` ~2 GiB, `gemma-3-270m` ~0.5 GiB, `gemma-3-4b-it` ~8.6 GiB,
`gemma-2-2b-it` ~5.2 GiB, `gemma-2-9b-it` ~18.5 GiB. Small totals; stage sequentially
per [`grid-per-sha-trees-fill-disk`](../workflow.md) (dgx disk at 95%). W0 accepts the
HF gate before fetching. Presence verified directly, not from a table (the OPT lesson).

**Blocking preconditions for the Gemma-4 row only:** (a) the 0.25.0 oracle must be able
to construct `Gemma4*` (unverified; probably absent — reopen when the oracle advances);
(b) a text-backbone checkpoint that fits the run headroom. Until both hold, W6 is an
honesty pass.

**Downward dependencies this introduces:** the shared `kGeluAndMul` op (reusable by any
GeGLU model); the final-logit-soft-cap in the logits path (reusable by any capped
model); the dual per-layer rope-cache selector.

### Work breakdown

- **W0 — Ground the facts on hardware.** Accept the HF gate; fetch `gemma-3-1b-it`
  (cheapest genuine text checkpoint); confirm the real `config.json` matches §0.5-0.6;
  run the pinned 0.25.0 oracle on it; run the K=5 greedy self-determinism probe that
  selects the gate form; **verify BOS placement vs the oracle** (the OPT lesson); and
  **probe whether 0.25.0 can construct `Gemma4*`** (decides the Gemma-4 row's fate). No
  code. *Gate: oracle reference outputs produced; determinism + BOS + Gemma-4-oracle
  verdicts recorded.* **LANDED 2026-07-24:** 0.25.0 constructs `Gemma3ForCausalLM` and
  runs it; `gemma-3-1b-it` config matches §0.5-0.6 (head_dim 256, `final_logit_softcapping:
  null` → the final soft-cap is NOT needed for the Gemma-3 gate); K=5 per-prompt greedy
  **ALL-DETERMINISTIC → STRICT bar** (1b self-deterministic, so per the ratified rule NO
  4b-MM-backbone check is required); **BOS verified** (all prompts prepend `bos_token_id=2`).
  (Gemma-4 oracle probe deferred to W6.)
- **W1 — Shared additive primitives.** `kGeluAndMul` (tanh-approx GeLU + mul) in
  `vt::ops` (CUDA + CPU), the `sqrt(hidden)` embed-scale, and the
  `query_pre_attn_scalar` attention scalar. Additive; no existing model sets them ->
  all 10 regressions byte-identical. *Gate: unit parity at real dims + regressions
  UNCHANGED.* **LANDED 2026-07-24:** `kGeluAndMul` + `kMulScalar` (bf16 embed-scale)
  CUDA+CPU appended before `OpId::kCount` (no op renumbered); qpas is a scalar threaded
  into the block (no op). Unit gate `test_ops_activation` **12/12** (GeGLU bit-exact vs
  in-test reference at I=6912 + MulScalar bf16).
- **W2 — Gemma-3 dense block + G1 gate.** New `gemma3.{h,cpp}`: sandwich norms (reuse
  glm4), QK-norm (reuse `kAttnQkNormRopeGate`), dual per-layer rope theta + interleaved
  sliding-window routing (reuse FA-2 window + KV specs), tied embeddings, W1 scalars.
  Reuses only the `dense_attn_block.h` glue + GQA paged path, per D4. *Gate: SACRED gate
  on gemma-3-1b-it (form per W0) + bigger strict on gemma-3-4b text backbone.* **LANDED
  2026-07-24:** `gemma3.{h,cpp}`/`gemma3_weights.cpp`/`gemma3_registry.cpp`. Note the q/k
  norm uses standalone `vt::RmsNorm{gemma=true}` (not the fused `kAttnQkNormRopeGate`
  recipe) + `RopeNeox` per-layer theta — the byte-identical deterministic default path.
  **SACRED gate STRICT token-exact 48/48** greedy vs vLLM 0.25.0 (tokenizer-free: feeds
  vLLM's exact BOS-prefixed ids into the CUDA prefill, mirroring the Mistral
  `LOAD-SENTENCEPIECE` blocker — our loader does not validate Gemma's byte_fallback BPE
  tokenizer.json). 1b self-deterministic ⇒ the 4b text-backbone strict check is NOT
  required (ratified rule). Loader 340 tensors, registry 23/23, clean `-Werror` 0 warn.
- **W3 — Final logit soft-cap.** Add `soft_cap` (tanh cap) to the logits path, plus the
  Gemma-2 attn soft-cap wiring through `Attention` -> FA-2. Default-inert for existing
  models. *Gate: unit parity + regressions UNCHANGED.* **LANDED 2026-07-24:** `vt::SoftCap`
  (`kSoftCap`, appended before `kCount`; final cap, monotone/greedy-invariant, unit
  bit-exact V=256000) + `PagedAttentionArgs.logits_soft_cap` (default 0.0) threaded into
  the NATIVE paged-attn score kernels (`cuda_paged_attn.cu`, `ApplySoftcap=cap*fast_tanh`
  via `tanh.approx.f32` mirroring flash-attn `cutlass::fast_tanh`), the CPU kernel, and the
  FA-2 fold. Reality vs the spike: the DEFAULT attention path is the NATIVE `cuda_paged_attn`
  kernels (FA-2 is env-gated OFF), so the soft-cap had to be threaded there (9 kernels), not
  just the FA-2 launcher; the MLA fa2 launcher (`MlaPrefillAttentionArgs`) is left untouched
  (MLA never soft-caps). cap==0 ⇒ byte-identical (regressions witness it).
- **W4 — `Gemma2ForCausalLM`.** New `gemma2.{h,cpp}` = the Gemma-3 block minus QK-norm,
  plus attn+final soft-cap and qpas. *Gate: SACRED on gemma-2-2b-it; STRICT on
  gemma-2-9b-it. Proves the soft-cap primitives.* **LANDED 2026-07-24:** `gemma2.{h,cpp}`/
  `_weights`/`_registry` on `unsloth/gemma-2-2b-it` (ungated bf16 mirror; `google/*`
  HF-gated). **near-tie-band SACRED 48/48** (vLLM K=5 ALL-DETERMINISTIC but 12/48 cells in its
  OWN 0.5-nat band; 44 strict + 4 at gap 0.0000 nats in vLLM's own logits, 0 forward-divergent
  — GLM-4 methodology). Soft-cap PROVEN applied (cap-on≠cap-off A/B + unit + CPU differs); the
  4 near-ties are kernel-path-invariant (a base bf16 near-tie). No 9b run (1b/2b self-det
  suffices per the ratified rule; disk). Loader 1427/288 tensors.
- **W5 — `GemmaForCausalLM` (Gemma 1).** Simplest; reuses W1's GeGLU + embed-scale.
  *Gate: SACRED on gemma-1.1-2b-it. Lowest priority, deferrable.* **LANDED 2026-07-24:**
  `gemma.{h,cpp}`/`_weights`/`_registry` on `unsloth/gemma-2b` (two fused norms/layer,
  head_dim scale, no soft-cap/QK-norm/sliding; `hidden_act:gelu`+no `hidden_activation` ⇒
  vLLM tanh-GeLU = our `kGeluAndMul`). **STRICT token-exact SACRED 48/48** (vLLM K=5
  ALL-DETERMINISTIC → STRICT; BOS-verified). Loader 815/164 tensors.
- **W6 — Gemma-4 characterization/honesty pass.** Record the 0.25.0 oracle-support
  verdict, config/registry resolution from the real MM `text_config`, and the
  PLE/YOCO/MoE/k_eq_v primitive inventory as a scoped follow-on campaign. Implement ONLY
  if W0 proved oracle support AND a text-backbone checkpoint fits. *Gate: gate 10.*
  **LANDED 2026-07-24 (honesty pass):** `test_gemma4_honesty` 6/6 — we do NOT register
  `Gemma4ForCausalLM` (resolving it fails loudly with the exact "not supported" oracle
  message; the 3 ported siblings resolve). 0.25.0 registry LISTS `Gemma4ForCausalLM` but
  every public checkpoint is multimodal-wrapped (≥12B); no bare text checkpoint. Recorded
  HW/DEP-BLOCKED with the PLE/YOCO/Gemma-4-MoE/k_eq_v stack; NO multimodal implementation.
- **W7 — Speed close.** Decode-graph sibling per landed Gemma + the binding every-axis
  grid vs vLLM. Nothing reaches `DONE` before this. *Gate: acceptance rule — match or
  beat vLLM on every axis.*

### Risks/decisions

**D1 — Gemma-4 leads the characterization but Gemma-3 is the first gate vehicle.** The
newest registered Gemma is Gemma-4 and it is real, but its only checkpoints are
multimodal-wrapped, its 0.25.0 oracle-support is unverified, and it needs a
PLE/YOCO/MoE/k_eq_v stack no other row uses. Mirroring how the GLM spike handled
`Glm4MoeForCausalLM` (newest, HW-blocked; gate vehicle = the one that fit), Gemma-3 is
the recent-first vehicle that is a clean, oracle-certain, fitting text bring-up.

**D2 — Do NOT extend `dense_attn_block.h::AttnBlock` for Gemma.** Gemma violates several
of its assumptions (sandwich norms, dual rope theta, per-layer sliding routing,
embed-scale/qpas scalars, GeGLU). The OPT + GLM precedent (their D4) already proved this
header does not stretch across families; reuse only the glue + GQA paged path, write a
new `gemma3.{h,cpp}` block. Tracked debt: a later block-consolidation pass, never
smuggled into a bring-up.

**D3 — The sandwich norm + BOS are silent-corruption hazards.** `gemma3.py:282,288`
applies standalone norms to the sublayer OUTPUT before the residual add, and Gemma
prepends BOS via the tokenizer. Getting either wrong emits fluent text while being
numerically wrong (the OPT 0/6 failure mode). Mitigation: gate 2 requires an e2e
token-exact gate for these, and W0 verifies BOS against the oracle before any forward
work.

**D4 — GeGLU numerics must match vLLM's `approximate="tanh"` exactly.** GeLU has two
variants (exact erf vs tanh approximation); Gemma uses the tanh approximation
(`gelu_pytorch_tanh`). A mismatch is a small, uniform, silent drift. Mitigation: unit-gate
`kGeluAndMul` bit-against vLLM's `GeluAndMul(approximate="tanh")` at real dims (gate 3),
and ground the constant `sqrt(2/pi)` and the cubic term against
`vllm/model_executor/layers/activation.py`.

**D5 — The Gemma-2 "hybrid attention" interleave is the load-bearing sliding-window
test.** Upstream labels the gemma-2-2b entry "test hybrid attention". Our sliding-window
kernel + KV specs exist, but the PER-LAYER routing (global vs local by `layer_types`,
each with its own rope theta and window) is new wiring. Mitigation: W2/W4 port the
upstream sliding-window correctness test and gate the interleave e2e, not just the
kernel in isolation.

**D6 — `Gemma3nForCausalLM` is out of scope by design, not omission.** It is the
MatFormer/AltUp/Laurel nano architecture — a distinct design with its own per-layer
embedding, activation sparsity and matryoshka nesting, not a Gemma-core delta. It stays
`INVENTORIED`; folding it in would misrepresent the Gemma-core bring-up as covering it.
