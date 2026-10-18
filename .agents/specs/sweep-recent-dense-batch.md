# SPIKE: recent-dense TEXT batch (Phi / Command-R / Granite / StableLM / InternLM2 / MiniCPM / Phi-3-4)

## TRIVIAL-TAIL IMPLEMENTATION UPDATE (2026-07-26, Yi + InternLM3 — base `origin/main` `4fff34f7`, worktree `sweep-yi-internlm3`, dgx `~/vllmcpp-yi-il3`). **CLOSES the recent-dense TEXT tier.**

Both remaining trivial-tail rows brought up as Llama aliases (each a near-zero-work reuse of
the landed Llama path). **W0 RUN-VERIFIED both on the pinned vLLM 0.25.0 oracle (the OLMo-3
lesson): both BUILD+RUN a coherent greedy golden**, unlike OLMo-3. Two SACRED gates PASS.

- **Yi (`YiForCausalLM`?) — VERDICT: modern Yi IS the Llama arch, ZERO code delta.** The
  checkpoints (`01-ai/Yi-1.5-6B-Chat`, `01-ai/Yi-Coder-1.5B-Chat`) declare
  `architectures:["LlamaForCausalLM"]` / `model_type:"llama"`, so Yi is ALREADY supported by
  the landed Llama registration — **no `YiForCausalLM` alias added** (the legacy arch id does
  NOT appear in modern checkpoints AND vLLM 0.25.0 registers no `YiForCausalLM` → mirror the
  oracle, add none). Gate vehicle `01-ai/Yi-Coder-1.5B-Chat` (ungated, 1.5B, distinct
  64000-vocab, rope_theta 1e7, GQA). **W0:** vLLM 0.25.0 builds+runs it coherently; per-prompt
  K=5 ALL-DETERMINISTIC ⇒ STRICT bar. **SACRED gate `test_yi_paged_engine` (dgx GB10): 16/16
  PASS — 13/16 STRICT token-exact + 3/16 near-tie band, max teacher-forced gap 0.125 nats, 0
  forward-divergent (140 assertions).** RED is vacuous (zero code delta); the Llama forward is
  separately STRICT-proven on `Llama-3.2-1B` (re-gated 16/16 UNCHANGED here). TEST + goldens
  only.
- **InternLM3 (`InternLM3ForCausalLM`) — VERDICT: a plain Llama alias in vLLM 0.25.0, NOT
  InternLM2+sliding-window.** The spike's trivial-tail note ("InternLM3 = InternLM2 + sliding
  window") was WRONG: `registry.py:134` maps `InternLM3ForCausalLM` → `("llama",
  "LlamaForCausalLM")`, and the `internlm3-8b-instruct` config carries NO `sliding_window` (it
  is RMSNorm + NeoX + GQA(kv=2) + SiLU-SwiGLU + **dynamic-NTK rope** factor 6.0 / theta 5e7,
  no biases, untied lm_head, head_dim 128, vocab 128512). **ONE registry ALIAS line** in
  `llama_registry.cpp` (`REGISTER_VLLM_MODEL(internlm3_llama, "InternLM3ForCausalLM",
  kLlamaFactory, kLlamaInfo)`) binds it to the landed Llama factory VERBATIM — zero
  forward/loader delta. Dynamic rope is already gate-proven (InternLM2's dynamic factor-2.0
  rides `DynamicNTKScalingRotaryEmbedding` and identity-within-window; InternLM3 factor 6.0 is
  the same path). **W0:** vLLM 0.25.0 builds+runs `internlm/internlm3-8b-instruct` coherently
  ("Paris.Paris is the capital of France…", "Albert Einstein…"); per-prompt K=5
  ALL-DETERMINISTIC ⇒ STRICT bar. **SACRED gate `test_internlm3_paged_engine` (dgx GB10): 16/16
  PASS — 14/16 STRICT token-exact + 2/16 near-tie band, max teacher-forced gap 0.0000 nats, 0
  forward-divergent (140 assertions).** RED (mutual-confirmation): the plain-Llama alias gates
  16/16 → InternLM3 does NOT need InternLM2's fused-`wqkv` interleaved de-split (which the
  InternLM2 row RED-proves is load-bearing for InternLM2); routing InternLM3 through that split
  would scramble its plain [q|k|v] weights. Alias + TEST + goldens only.
- **Tokenizer (both, additive vehicle prep):** Yi and internlm3 ship a SentencePiece
  `tokenizer.model` whose HF form is either absent (`tokenizer.model` only) or a null
  pre_tokenizer + `Replace " "→"▁"` normalizer that our parser rejects
  (`tokenizer.cpp:293,355`). Gated ID-based via the `TokensPrompt` path (feeds the oracle's
  exact prompt ids). `scripts/stage-tokenizer-metaspace.py` (reproducible, gate-neutral)
  generates a BPE `tokenizer.json` (fast-backend for Yi; direct SentencePiece→BPE extraction
  for internlm3's custom slow `InternLM3Tokenizer`) and re-expresses whitespace as the
  equivalent Metaspace pre_tokenizer our parser accepts (MiniCPM precedent). Only affects
  engine construction + detok display; the forward is ID-gated.
- **Non-regression:** additive only — `git diff --stat` = one `llama_registry.cpp` alias line +
  comment, two new tests, CMake, staging script, goldens, records. No shared op/kernel touched
  ⇒ every other model byte-identical. `Llama-3.2-1B` SACRED re-gated **16/16 UNCHANGED** (I
  touched `llama_registry.cpp`). Clean CUDA `-Werror` 0 warnings; no kernel touched (no
  compute-sanitizer needed). **SPEED PENDING** (both eager).
- **★ TIER STATUS: the recent-dense TEXT tier is CLOSED.** All 8 families accounted for:
  Granite-3 / StableLM / InternLM2 / Phi-3-4 / Phi-1-2 / MiniCPM / MiniCPM3 SACRED-landed;
  Command-R HF-gate-blocked (implemented, no ungated real vehicle); OLMo-3 oracle-pin-blocked
  (implemented, vLLM 0.25.0 can't run it); + Yi (Llama arch, zero delta) and InternLM3 (Llama
  alias) SACRED-landed here. No trivial-tail rows remain.

## RANK-9 IMPLEMENTATION UPDATE (2026-07-26, MiniCPM3 — base `origin/main` `a8363c60`, worktree `sweep-minicpm3`, dgx `~/vllmcpp-minicpm3`)

- **rank 9 MiniCPM3 (`MiniCPM3ForCausalLM`, `openbmb/MiniCPM3-4B`) — SACRED 16/16, row `ACTIVE`. CLOSES the non-trivial recent-dense tier** (only the trivial tail — Yi=Llama-alias, InternLM3=InternLM2+sliding-window — would remain).
  **ZERO NEW compute KERNEL** as scoped — MiniCPM3 is the landed MiniCPM 3-scalar dense
  skeleton with attention swapped GQA→**MLA**, REUSING the landed DeepSeek-V2 MLA block
  (`mla::ForwardMlaAttentionBlock`, load-time `kv_b_proj`→W_UK/W_UV absorption) threaded
  with MiniCPM3 dims. Grounded in `minicpm3.py` @ `e24d1b24`: `MiniCPM3Attention` MLA
  geometry (`:52-134`, q_a_proj/q_a_layernorm/q_b_proj + kv_a_proj_with_mqa/kv_a_layernorm/
  kv_b_proj + o_proj), `MiniCPM3DecoderLayer`/`MiniCPM3Model`/`MiniCPM3ForCausalLM`
  subclass MiniCPM (`:186-233`) so the 3 scalars (scale_emb 12, scale_depth/sqrt(62),
  hidden/scale_width=2560/256=10) are inherited verbatim from the landed `minicpm.cpp`.
- **THREE MLA deltas vs the landed DeepSeek-V2 path**, all handled faithfully:
  (1) **`is_neox_style=True`** — MiniCPM3 takes get_rope's default neox rotation
  (`:121-125`), DeepSeek uses gptj (`is_neox_style=False`). Threaded through the shared
  `mla::MlaBlockDims::is_neox_style` (DEFAULT false → every DeepSeek/GLM registration is
  byte-identical); the MLA block's decoupled-rope call now reads `dims.is_neox_style`.
  (2) **LongRoPE not YaRN** — MiniCPM3-4B ships `rope_scaling type=longrope`; since
  max_pos==original_max_pos (32768) the LongRoPE scale is 1.0 ⇒ mscale 1.0, the SHORT
  cache is selected, and the softmax scale is the plain qk_head_dim**-0.5 (NO mscale^2
  correction). A small `BuildMiniCPM3RopeCosSinCache` (phi3_long_rope_scaled_rope.py:97-123)
  replaces `BuildDeepseekRopeCosSinCache`. (3) q_lora ALWAYS present → only the
  fused_qkv_a_proj branch runs.
- **ONE shared-kernel change (reuse, not a new kernel):** the FA-2 MLA prefill only had
  hdim {192,256} instantiated; MiniCPM3's qk_head_dim is 96. The caller
  (`cuda_mla_prefill.cu`) now rounds qk_head_dim UP to the nearest compiled FA-2 dim and
  zero-pads Q/K/V into it (96→128, reusing the already-compiled `hdim128` split-KV kernel);
  the dispatch gate (`cuda_flash_attn_fa2.cu`) accepts d=128. EXACT: trailing zeros add
  nothing to the QK dot (scale passed explicitly) and V's padded output columns are sliced
  off. For DeepSeek (d=192)/GLM (d=256) `d_pad==dqk` so Q/K pass native and only V is
  padded — byte-identical to before.
- **W0 (RUN-VERIFIED):** `openbmb/MiniCPM3-4B` is ungated but ships ONLY `pytorch_model.bin`
  (no safetensors); converted `.bin`→safetensors via trusted torch on the oracle box
  (`scripts/minicpm3-convert-safetensors.py`; tied embeddings — no lm_head — bf16, 746
  tensors). vLLM 0.25.0 BUILDS+RUNS `MiniCPM3ForCausalLM` with `trust_remote_code=True`
  (coherent greedy ~19 tok/s). K=5 self-determinism: ALL 16/16 prompts DETERMINISTIC → the
  gate holds us to STRICT. MLA config: qk_nope 64, qk_rope 32, qk_head_dim 96, v_head_dim
  64, kv_lora 256, q_lora 768, MLA cache head_size 288, rope_theta 10000, longrope.
- **W4 SACRED (`test_minicpm3_paged_engine`, dgx):** 16/16 PASS (13/16 STRICT token-exact +
  3/16 near-tie band only; max teacher-forced root-divergence gap 0.0000 nats across all 20
  divergent positions — perfect bf16 ties; 0 forward-divergent; 140 assertions). Gated via
  oracle prompt ids (TokensPrompt path). **RED:** flipping `is_neox_style` to the wrong
  (DeepSeek gptj) rotation → first-token divergence, gate FAILS loudly. **DeepSeek-V2-Lite
  non-regression:** `test_deepseek_v2_paged_engine` re-gated 8/8 (unchanged — the is_neox
  default + identity prefill padding leave it byte-identical). CUDA `-Werror` clean;
  compute-sanitizer memcheck 0 on the padded prefill. SPEED PENDING (eager; the decode
  CUDA-graph sibling of DeepSeek-V2 W9 is the follow-up).

## RANK-5 IMPLEMENTATION UPDATE (2026-07-26, MiniCPM — base `origin/main` `c39d78a6`, worktree `minicpm-bringup`, dgx `~/vllmcpp-minicpm`)

- **rank 5 MiniCPM (`MiniCPMForCausalLM`, `openbmb/MiniCPM-2B-sft-bf16`) — SACRED 16/16, row `ACTIVE`.**
  **ZERO NEW KERNEL** as scoped — MiniCPM is the landed Granite forward with three
  scalar deltas (grounded in `minicpm.py` @ `e24d1b24`, reusing the shared dense glue):
  scale_emb=12 after embed (`:441-443` `MiniCPMModel.embed_input_ids`, `vt::MulScalar`);
  scale_depth/sqrt(num_layers)=1.4/sqrt(40)=0.2214 scaled residual add on BOTH attn+mlp
  sublayer outputs (`:384-386,392-393` `MiniCPMDecoderLayer.forward`, `vt::MulScalar`+
  `vt::Add`, NON-fused residual); hidden divided by scale_width=hidden_size/dim_model_base
  =2304/256=9.0 before lm_head (`:604,633,640`, `vt::MulScalar` on the normed hidden).
  Standard 1/sqrt(head_dim) attn scale (MiniCPM has NO custom attention multiplier — the
  one delta vs Granite). Tied lm_head (checkpoint has no lm_head.weight). New files only
  (`minicpm.{h,cpp}`, `minicpm_weights.cpp`, `minicpm_registry.cpp`,
  `test_minicpm_paged_engine.cpp`) + one `REGISTER_VLLM_MODEL`.
  - **W0 RUN-VERIFIED + `.bin` RISK RESOLVED (spec §0.3 MEDIUM oracle-risk, D5).** vLLM
    0.25.0 BUILDS+RUNS `MiniCPMForCausalLM` with `trust_remote_code=True` (coherent greedy;
    neither transformers nor vLLM register the `minicpm` config → trust_remote_code loads
    `configuration_minicpm.MiniCPMConfig`; vLLM still uses its OWN `minicpm.py`). The ONLY
    checkpoint format is `pytorch_model.bin` (NO safetensors, our loader is safetensors-only)
    → resolved WITHOUT a pickle loader: converted the OFFICIAL openbmb `.bin`→safetensors via
    TRUSTED torch on the oracle box (`scripts/minicpm-convert-safetensors.py`), so BOTH the
    vLLM golden AND our engine read IDENTICAL bf16 weights. Ungated. Tied embeddings, scalars
    scale_emb=12/scale_depth=1.4/dim_model_base=256 confirmed against config.json.
  - **Gate form BY MEASUREMENT:** vLLM per-prompt K=5 **ALL-DETERMINISTIC** (0 multi-member
    cells) ⇒ STRICT well-posed; **16/16 PASS** by the ratified near-tie ROOT-divergence gate
    — **10/16 STRICT token-exact + 6/16 near-tie band, max teacher-forced gap 0.0000 nats, 0
    forward-divergent** (the near-ties are vLLM's own prefill/decode inconsistency cascading —
    e.g. vLLM greedy degenerated into "-"×10 on p14 while every one of our tokens IS vLLM's
    teacher-forced argmax at gap 0.0). RED-first: dropping scale_depth (residual_scale→1.0) →
    **256/256 positions divergent, max gap 29.375 nats** the gate CATCHES — scale_depth is
    load-bearing. Clean CUDA `-Werror` 0 warnings; no kernel touched.
  - **Tokenizer (D4):** MiniCPM's tokenizer.json encodes SentencePiece whitespace as a
    normalizer Sequence [Prepend "▁", Replace " "→"▁"] with null pre_tokenizer, which our
    parser rejects. Gated via the additive `TokensPrompt` engine path (InternLM2/Command-R
    precedent) feeding the oracle's exact prompt ids; the vehicle tokenizer.json is
    faithfully re-expressed as a Metaspace pre_tokenizer for engine construction/detok. A
    native normalizer Prepend/Replace port is the orthogonal follow-up. SPEED PENDING.
    Remaining rows (Command-R, MiniCPM3) stay `SPIKE`, one agent each.

## RANK-8 IMPLEMENTATION UPDATE (2026-07-26, Phi-1/2 — base `origin/main` `29c710dd`, worktree `phi12-bringup`, dgx `~/vllmcpp-phi12`)

- **rank 8 Phi-1/Phi-2 (`PhiForCausalLM`, `microsoft/phi-2`) — SACRED 16/16, row `ACTIVE`.**
  The OLDER Microsoft Phi arch, DISTINCT from the landed `Phi3ForCausalLM`. **W0 RUN-verified**
  the oracle (builds+runs coherent greedy text; ungated + safetensors; ~5.18 GiB F16). Gate form
  BY MEASUREMENT: vLLM 0.25.0 per-prompt K=5 **ALL-DETERMINISTIC** (0 multi-member cells) ⇒
  STRICT well-posed; **16/16 PASS** by the ratified near-tie ROOT-divergence gate — **9/16
  STRICT token-exact + 7/16 near-tie band, max teacher-forced gap 0.25 nats < 0.5, 0
  forward-divergent** (every downstream token == vLLM's teacher-forced argmax at gap 0.0).
  - **ZERO-NEW-KERNEL — spec §0.2 row 7 CORRECTED.** The spike predicted "ONE new op: a
    `kGelu`/NewGELU unary". WRONG: `gelu_new` (activation.py:516-519, NewGELU
    `0.5x(1+tanh(√(2/π)(x+0.044715x³)))`) is **bit-identical to the landed `vt::GeluTanh`**
    (the Qwen3-VL vision-tower unary) — the non-gated MLP reuses it, so **NO new op**. New
    files only (`phi.{h,cpp}`, `phi_weights.cpp`, `phi_registry.cpp`, `test_phi_paged_engine.cpp`).
  - **Deltas ported (all REUSE), grounded in `phi.py` @ `e24d1b24`:** GPT-J PARALLEL residual
    (`PhiLayer.forward` :189-202: ONE nn.LayerNorm+bias feeds BOTH attn+mlp, outputs summed
    together — reuses the Command-R `RunLayer` wiring); biased q/k/v/`dense` (`:98,102`, OPT
    `BiasedProj`); partial NeoX RoPE (rotary_dim 32/head_dim 80, `partial_rotary_factor` 0.4,
    `RopeFromCache` — the phi3/StableLM path); non-gated NewGELU MLP (`PhiMLP` :165-169, fc1->
    `vt::GeluTanh`->fc2, both biased); untied lm_head WITH per-vocab bias (`:288-320`, `vt::Matmul`
    + f32 `vt::Add` broadcast).
  - **F16 checkpoint (new fact, spec §0.3 said ~5.6 GiB bf16).** microsoft/phi-2 is **FLOAT16 on
    disk (all 453 tensors F16)**; vLLM serves it bf16 (`Casting torch.float16 to torch.bfloat16`).
    The loader is DTYPE-AWARE: BF16 reuses the shared helpers verbatim, F16 downcasts f16->f32->
    bf16(RNE) — bit-identical to torch `.to(bfloat16)`. Kept LOCAL to `phi_weights.cpp` (mirrors
    OLMo-2's F32→BF16 converters); the shared `dense_weight_loaders.h` is UNTOUCHED ⇒ every other
    model byte-identical.
  - **RED-first (3 load-bearing deltas):** drop qkv bias → max gap **1.25 nats** (gate CATCHES);
    sequential-instead-of-parallel residual → max gap **21.19 nats** (gate CATCHES); wrong partial-
    rotary fraction (full 80 or half 16) → hard `rope_from_cache` shape-contract abort (cannot
    silently run — structurally caught). memcheck 0 errors. Clean CUDA `-Werror` 0 warnings; no
    kernel touched. SPEED PENDING. Remaining rows (MiniCPM, MiniCPM3) stay `SPIKE`.

## TOP-3 IMPLEMENTATION UPDATE (2026-07-24, batch3 — `sweep-recent-dense-batch3`, base `origin/main` `e1173fa`, dgx `~/vllmcpp-b3`)

The three §0.6 recommendations were implemented, one clean `-Werror` build, one
regression pass. **All three ZERO-NEW-KERNEL as scoped** (Granite reuses the shared
dense glue + `vt::MulScalar`/`vt::Add`; Phi-3 reuses the pinned `Phi3LongRoPE` CPU
rope class + `RopeFromCache`; OLMo-3 reuses the yarn CPU rope class + the Gemma-3
sliding-window arg). Results:

- **rank 3 Granite-3 (`GraniteForCausalLM`, `granite-3.3-2b-instruct`) — SACRED 16/16, row `ACTIVE`.**
  vLLM K=5 ALL-DETERMINISTIC → STRICT; 15/16 token-exact + 1 near-tie 0.062 nats, 0
  forward-divergent. The 4 default-1 scalars + the attention scale (0.015625=1/64,
  NOT 1/sqrt(64)) proven. memcheck 0 errors. New files only.
- **rank 2 Phi-3/Phi-4 (`Phi3ForCausalLM`, `Phi-4-mini-instruct`) — 15/16, row `GATING`.**
  Pre-fused qkv/gate_up loader + LongRoPE (partial rotary 96/128, long_factor+mscale)
  fed by real positions. 7/16 token-exact + 8 near-tie + 1 forward-divergent (p12tok6
  1.0 nats > 0.5 band) — coherent output, structurally correct; the 1 residual is a
  LongRoPE bf16-cache precision follow-on. Forced two guarded, diff-inert shared-TU
  fixes (regression-witnessed): `hf_config.cpp` longrope `original_max_position_embeddings`
  top-level fallback (Phi-4 stores it top-level, D4 anchor) + `tokenizer.cpp` accept
  lstrip/rstrip added-token options (Phi-4 `<\|assistant\|>`).
- **rank 1 OLMo-3 (`Olmo3ForCausalLM`, `OLMo-3-1025-7B`) — IMPLEMENTED, oracle-BLOCKED (D5).**
  Rides the landed olmo2 row via guarded additive edits (diff-inert; OLMo-2 gate 16/16
  UNCHANGED): per-layer dual rope (plain sliding theta 500000 vs YaRN full-attn, routed
  by `config.layer_types`) + finite sliding window (inert for short gate contexts) +
  dtype-aware BF16 loader. **The pinned vLLM 0.25.0 oracle CANNOT run the checkpoint**
  (`olmo2.py:143` `rope_parameters["rope_theta"]` → `KeyError`; per-layer-type rope
  schema newer than the oracle's transformers), so there is NO SACRED bar (exactly the
  D5 "no oracle → honestly blocked" case). Our engine loads + runs it; W5 gate pending
  a pin/oracle advance. The §0.5/§0.6 D5 caveat (oracle-support must be probed) is thus
  confirmed for OLMo-3, not just MiniCPM/pin-removed names.

Regressions byte-identical (OLMo-2 16/16 · Qwen3-dense 184/184 · OPT 63/63 · Llama
92/92 · Mistral 92/92 re-run; big gates by construction). SPEED PENDING for all three.
Remaining rows (MiniCPM, InternLM2, Command-R, Phi-1/2, MiniCPM3) stay
`SPIKE`, one agent each per the queue below.

## RANK-6 IMPLEMENTATION UPDATE (2026-07-26, InternLM2 — base `origin/main` `43287971`, worktree `internlm2-bringup`, dgx `~/vllmcpp-internlm2`)

- **rank 6 InternLM2 (`InternLM2ForCausalLM`, `internlm2-chat-1_8b`) — SACRED 16/16, row `ACTIVE`.**
  **W0 RUN-verified** the oracle (builds+runs coherent greedy text; ungated + safetensors
  + ~3.8 GiB). **ZERO new kernel** as scoped: `internlm2.h` aliases the landed
  Llama/Qwen3-dense forward VERBATIM (`using InternLM2Model = Qwen3DenseModel`); the ONLY
  delta is the LOADER de-interleave of the fused `wqkv` (packed q/k/v INTERLEAVED by
  kv-group, `internlm2.py:158-176 split_qkv`) into the plain [q|k|v]-row merged qkv_proj
  the shared AttnBlock consumes. rope_scaling `dynamic` is identity here (base unchanged).
  vLLM K=5 ALL-DETERMINISTIC → STRICT; 12/16 token-exact + 4/16 near-tie band, **max gap
  0.0000 nats**, 0 forward-divergent (the 4 near-ties are EOS-overrun — vLLM stops on its
  generation_config secondary eos 92542, our engine reads only config.json eos=2 — plus
  vLLM prefill/decode tie self-inconsistency; every teacher-forced argmax == our token).
  **RED wrong-split (non-interleaved concat) CAUGHT**: engine emits 55040 vs 1934 at
  prompt0/tok0, gate FATAL — the interleave is load-bearing. New TUs only + an ADDITIVE
  engine `TokensPrompt` path (`LLMEngine::{add_request,generate}` /
  `InputProcessor::process_inputs_tokens`, string path byte-identical) to gate on the
  oracle's exact prompt ids — InternLM2's non-standard fast BPE (no pre_tokenizer,
  fuse_unk drops spaces) is not in our tokenizer families; a full port is orthogonal/out
  of scope. Clean CUDA `-Werror` 0 warnings; no kernel touched. SPEED PENDING. Remaining
  rows (MiniCPM, Command-R, Phi-1/2, MiniCPM3) stay `SPIKE`, one agent each.

## RANK-4 IMPLEMENTATION UPDATE (2026-07-26, StableLM — base `origin/main` `fb7609f7`, dgx `~/vllmcpp-stablelm`)

**rank 4 StableLM (`StableLmForCausalLM`, `stablelm-2-1_6b`) — SACRED 16/16, row `ACTIVE`.**
ZERO-NEW-KERNEL as scoped: `nn.LayerNorm` (weight+bias, NON-fused explicit residual —
REUSE the OPT-landed `vt::LayerNorm`) + partial NeoX RoPE (rotary_dim 16 of head_dim 64,
`partial_rotary_factor` 0.25, plain `default` cache built via the landed `RotaryEmbedding`
class — REUSE the phi3/GLM-4 `vt::RopeFromCache` partial path) + merged qkv **bias**
(`use_qkv_bias`=True — REUSE `LoadMergedBf16Vector` + `vt::Add` row-broadcast, the OPT
biased-projection path) + SiLU-SwiGLU (REUSE). New files only (`stablelm.{h,cpp}`,
`stablelm_weights.cpp`, `stablelm_registry.cpp`, `test_stablelm_paged_engine.cpp`); the
forward is written fresh (NOT `dense_attn::AttnBlock`) exactly like OPT/phi3 because the
non-fused LayerNorm residual + merged-qkv-bias cannot ride that block.

**W0 RUN-VERIFIED** (not config-construct): the pinned vLLM 0.25.0 oracle BUILDS+RUNS
`stablelm-2-1_6b` (ungated, ~3.3 GiB, fits). Gate form BY MEASUREMENT: per-prompt
(batch=1) K=5 greedy is **ALL-DETERMINISTIC** (0 multi-member cells) ⇒ STRICT bar (like
Granite); W0's BATCHED 24-token run showed one bf16 near-tie flip, so the ratified near-tie
harness is retained. Gate: **14/16 STRICT token-exact + 2/16 near-tie-band, max
teacher-forced gap 0.438 nats @ prompt[13] tok0, 0 forward-divergent**. RED-first: disabling
the qkv bias → prompt0 tok0 emits 4697 vs golden 264 (gate FAILS) — the `use_qkv_bias`
delta is load-bearing. memcheck 0 errors; clean CUDA `-Werror` 0 warnings.

**One shared-file touch (additive, proven inert):** `stablelm-2`'s Arcade100k tokenizer
stores the cl100k/Qwen2 split regex with LITERAL CR/LF control chars where the Qwen
checkpoints use `\r`/`\n` escapes (same regex). `tokenizer.cpp::DetectPattern` now
canonicalizes literal CR/LF → escape form before its equality checks (a no-op for every
checkpoint already using escapes). Inertness: `test_pretokenizer` 97926/97926 +
`test_tokenizer_parity` 1175/1175 unchanged. SPEED PENDING.

---

**SPIKE ONLY — no implementation, no kernels, no build, no benchmark, nothing
downloaded (only HF API metadata read).** A BATCH-SCOPING triage of the next tier
of recent dense (and small-MoE) text families, so the sweep has a RANKED, GROUNDED
implementation queue and near-additive families can be brought up rapidly, one impl
agent each, OLMo-2-style. Design + records only.

**Base:** `origin/main` `5c00fc4` (OLMo-2 `Olmo2/Olmo3ForCausalLM` W0-W4 landed).
**Oracle pin:** `/home/mudler/_git/vllm` @ `e24d1b24`. **dgx oracle:**
`~/venvs/vllm-oracle` = vLLM **0.25.0**. **Claim:** `CLAIM-SWEEP-RECENT-DENSE`.
**Parent plan:** [`breadth-sweep-plan.md`](breadth-sweep-plan.md) §B (the ranked
model sweep). **Gold-standard spike shapes mirrored:** [`sweep-olmo2.md`](sweep-olmo2.md)
(the ZERO-new-kernel dense bring-up pattern), [`sweep-gemma.md`](sweep-gemma.md)
(the multi-row family batch + per-version delta table).

Rows advanced `INVENTORIED` -> `SPIKE` by this spike (8):
`MODEL-TEXT-phi3-phi3-for-causal-lm`,
`MODEL-TEXT-granite-granite-for-causal-lm`,
`MODEL-TEXT-stablelm-stablelm-for-causal-lm`,
`MODEL-TEXT-minicpm-mini-cpmfor-causal-lm`,
`MODEL-TEXT-internlm2-intern-lm2-for-causal-lm`,
`MODEL-TEXT-commandr-cohere-for-causal-lm`,
`MODEL-TEXT-phi-phi-for-causal-lm`,
`MODEL-TEXT-minicpm3-mini-cpm3-for-causal-lm`.

Called out but NOT a new row here (each near-free on a LANDED row): **OLMo-3**
(`Olmo3ForCausalLM`, rides the ACTIVE `MODEL-TEXT-olmo2-olmo2-for-causal-lm` row as
its W5 sliding-window follow-on — the cheapest next win) and **InternLM3 / Yi**
(both map to the ACTIVE `MODEL-TEXT-llama-llama-for-causal-lm` row — registry-alias
additions, not separate architectures).

Left `INVENTORIED` (characterized below with a reason, deprioritized): Falcon
(`FalconForCausalLM`, older + alibi/parallel branches), Falcon-H1
(`FalconH1ForCausalLM`, Mamba2 SSM hybrid — campaign), GraniteMoe /
GraniteMoeShared / GraniteMoeHybrid, Cohere2Moe (`Cohere2MoeForCausalLM`), PhiMoE
(`PhiMoEForCausalLM`) — all MoE/SSM campaigns. And the pin-REMOVED names (see §0.4).

---

## 0. Headline findings

### 0.0 The batch is dominated by ZERO-NEW-KERNEL near-additives over the LANDED substrate

The dense + LayerNorm + scalar-multiplier + partial-rope + MLA infrastructure that
Qwen3/Llama/Mistral/OPT/GLM/Gemma/OLMo-2/DeepSeek-V2 already landed covers most of
this tier with **new files only**. Of the 8 newly-scoped rows, **4 are
ZERO-NEW-KERNEL near-additive** (Phi-3/Phi-4, Granite-3, StableLM, MiniCPM), and two
more zero-new-kernel bring-ups are available off LANDED rows (OLMo-3 W5; InternLM3/Yi
Llama aliases) — **7 zero-new-kernel bring-ups total**. The 4 `small-new-op` rows add
at most ONE small thing each (a plain non-gated GELU unary for Phi-1/2; a scalar +
parallel-residual wiring for Command-R; a fused-`wqkv` interleaved split for InternLM2;
MLA re-wiring for MiniCPM3). No row in this batch needs a genuinely-new compute kernel
of the fp4/GDN/attention class.

### 0.1 The landed primitive inventory this batch reuses (verified in `src/`/`include/` @ `5c00fc4`)

- **Plain RMSNorm** `vt::RmsNorm {gemma=false}` (`include/vt/recipes.h`, `ops.h:90`) —
  Phi-3/4, Granite, MiniCPM/3, InternLM2.
- **`nn.LayerNorm` (mean+variance, weight AND optional bias)** — `vt::LayerNorm`
  (`ops.h:181`, `include/vt/ops.h:1215`), landed for OPT. Covers StableLM (weight+bias
  optional), Phi-1/2 (`input_layernorm`/`final_layernorm`), and Command-R's Cohere
  `LayerNorm` (no-bias -> null bias pointer).
- **Partial NeoX RoPE** (`rotary_dim < head_dim`) — the partial in-place + from-cache
  paths (`ops.h:1243,1249,1310`), landed for GLM-4 decoupled rope. Covers StableLM +
  Phi-1/2 partial rotary, and Phi-3 su/longrope's partial application if configured.
- **SiLU-SwiGLU** `kSiluAndMul` (`ops.h:91`) — Phi-3/4, Granite, StableLM, MiniCPM,
  InternLM2 MLP.
- **`kMulScalar`** (bf16 scalar multiply, `ops.h:201`, landed for Gemma embed-scale) —
  the Granite/MiniCPM scalar multipliers ride this + folded scalars.
- **MLA / latent-KV attention** (`DeepseekV2*` block, q_lora/kv_lora, decoupled rope,
  head_dim-256 prefill; landed W6 for DeepSeek-V2-Lite + GLM-4.7-Flash) — MiniCPM3.
- **Interleaved sliding-window / local+global attention** (FA-2 finite window +
  `SlidingWindowSpec`/`ChunkedLocalAttentionSpec` + per-layer `layer_types` routing,
  landed for Gemma-3 / OLMo-3) — Command-R (Cohere2/v2) + OLMo-3.
- **Merged-column loader** (`qkv_proj`, `gate_up_proj` packed_modules_mapping),
  **tied-or-untied embeddings**, the **shape-agnostic runner**, the
  `REGISTER_VLLM_MODEL` seam, the decode-graph sibling pattern — all reused.
- **Tokenizers:** ByteLevel BPE (Isolated + Removed-inverted) covers Granite /
  StableLM / Command-R / Phi-2 / Phi-4; SentencePiece(partial) covers Phi-3 / MiniCPM /
  MiniCPM3 / InternLM2 (same `LOAD-SENTENCEPIECE` path landed for Mistral/Gemma). BOS
  handling is a per-family W0 verify (the OPT/Llama trap: a silently mis-placed BOS
  scores 0/n while emitting fluent text).

### 0.2 The genuinely-NEW delta per family (small, additive)

| # | Family | Row | Dense/MoE/hybrid | The specific NEW delta over LANDED | Class |
|---|---|---|---|---|---|
| 1 | **Phi-3 / Phi-4** | `phi3-phi3-for-causal-lm` | dense | `Phi3ForCausalLM` is a `LlamaForCausalLM` subclass (`phi3.py:10`) with pre-fused `qkv_proj`/`gate_up_proj` checkpoints; only delta is optional su/longrope rope-scaling on long-context variants (base 4k = plain NeoX). Reuses the DONE Llama forward VERBATIM. | **ZERO-NEW-KERNEL** |
| 2 | **Granite-3** | `granite-granite-for-causal-lm` | dense | Llama + FOUR scalar multipliers (`embedding_multiplier` `granite.py:313`, `residual_multiplier` `:240,245`, `attention_multiplier`->scaling `:137`, `logits_scaling` `:371-372`); tied lm_head. All default-1 scalars threaded like Gemma qpas/embed-scale. | **ZERO-NEW-KERNEL** |
| 3 | **StableLM** | `stablelm-stablelm-for-causal-lm` | dense | `nn.LayerNorm` (not RMS; REUSE `kLayerNorm`) + partial rotary (`partial_rotary_factor`; REUSE partial NeoX) + optional `use_qkv_bias` (REUSE biased qkv) + SiLU-SwiGLU; standard pre-norm (separate input/post norms `stablelm.py:190-191`). | **ZERO-NEW-KERNEL** |
| 4 | **MiniCPM** | `minicpm-mini-cpmfor-causal-lm` | dense (MoE variant gated off) | Llama + `scale_emb` embedding scale (`minicpm.py:443`), `scale_depth/sqrt(num_layers)` residual scaling (`:384-385,392-393`), `dim_model_base` logit scaling. All scalars. Tied. (The `MiniCPMMoE` class `:82` is config-gated; the dense checkpoint is scalar-only.) | **ZERO-NEW-KERNEL** |
| 5 | **InternLM2** | `internlm2-intern-lm2-for-causal-lm` | dense | RMSNorm+NeoX+SiLU pre-norm = Llama, BUT the fused `wqkv` packs q/k/v INTERLEAVED by kv-group (`split_qkv` `internlm2.py:158-188`) — a LOADER/split-layout delta only, NO compute kernel. `w1/w3 -> gate_up`. | **small-new-op** (loader layout) |
| 6 | **Command-R / Command-R7B** | `commandr-cohere-for-causal-lm` | dense (one file for Cohere+Cohere2) | `logit_scale` on the logits proc (`commandr.py:376`); Cohere `LayerNorm` (no learnable bias, mean-centred -> REUSE `kLayerNorm` null-bias, verify mean-subtract at W0); **parallel-residual** block (single input norm, attn+MLP both off the normed input, summed `:264-272`); Cohere2/v2 adds per-head QK-`LayerNorm` (`:200-213`) + interleaved sliding window (`:184-197`, REUSE Gemma-3) + rope-on-sliding/v1-only; tied embeds asserted (`:372`), no biases. | **small-new-op** (logit_scale scalar + parallel-residual wiring + per-head-LN qk-norm) |
| 7 | **Phi-1 / Phi-2** | `phi-phi-for-causal-lm` | dense | Parallel residual (attn + mlp + residual, single input `nn.LayerNorm` `phi.py:201`; REUSE `kLayerNorm`), partial rotary (REUSE), qkv **bias**=True (`:98`; REUSE), **plain non-gated GELU MLP** (`fc1`->NewGELU->`fc2` `:166-168`) — the ONE new op: a `kGelu`/NewGELU UNARY (we only have gated `kGeluAndMul`); lm_head **bias** + UNTIED (`:279,291`). | **small-new-op** (NewGELU unary) |
| 8 | **MiniCPM3** | `minicpm3-mini-cpm3-for-causal-lm` | dense + MLA | MiniCPM scalars (subclasses `MiniCPMDecoderLayer` `minicpm3.py:186`) + **MLA attention** (q_lora/kv_lora, `q_a_layernorm`/`kv_a_layernorm`, decoupled rope `:52-156`) — REWIRES the LANDED DeepSeek-V2 MLA block; no new kernel. | **small-new-op** (MLA re-wire, reuses campaign) |

### 0.3 Hardware fit + tokenizer + oracle-risk (HF API 2026-07-24, metadata only, nothing downloaded)

bf16 on disk ~= 2 x params. GB10: ~119 GiB unified pool, ~113 GiB free now — every
smallest genuine checkpoint fits with room for a build tree. Presence is NOT read from
this table (the OPT lesson): W0 verifies weight FILES on dgx before a row is picked up.

| Family | Smallest genuine checkpoint | Params | bf16 on disk | Gated? | Tokenizer + BOS note | Oracle-support risk (0.25.0) |
|---|---|---|---|---|---|---|
| **Phi-3 / Phi-4** | `microsoft/Phi-4-mini-instruct` | 3.836B | ~7.7 GiB | ungated | Phi-4 = ByteLevel BPE (o200k-ish); Phi-3-mini = SentencePiece (Llama, 32064); BOS varies (Phi-3 `<s>`; Phi-4 `<\|endoftext\|>`) — W0 verify. Both REUSE. | LOW — `Phi3ForCausalLM` well-established; bigger STRICT `microsoft/phi-4` 14.66B (~29.3 GiB) also `Phi3ForCausalLM` |
| **Granite-3** | `ibm-granite/granite-3.3-2b-instruct` | 2.53B | ~5.1 GiB | ungated | ByteLevel BPE (StarCoder/GPT-2 family, 49k); BOS W0-verify. REUSE. | LOW — Granite well-established |
| **StableLM** | `stabilityai/stablelm-2-1_6b` | 1.64B | ~3.3 GiB | ungated | Arcade100k GPT-NeoX ByteLevel BPE. REUSE. | LOW; `stablelm-3b-4e1t` 2.8B a bigger strict |
| **MiniCPM** | `openbmb/MiniCPM-2B-sft-bf16` | ~2.7B | ~5.4 GiB | ungated | Llama SentencePiece. REUSE. NOTE: `.bin` (no safetensors), needs `trust_remote_code`. | MEDIUM — needs `trust_remote_code=True`; W0 confirms the 0.25.0 oracle constructs it |
| **InternLM2** | `internlm/internlm2-chat-1_8b` | 1.89B | ~3.8 GiB | ungated | SentencePiece. REUSE. | LOW |
| **Command-R7B** | `CohereLabs/c4ai-command-r7b-12-2024` | 8.03B | ~16 GiB | **gated:auto** (HF click-through) | Cohere ByteLevel BPE; `<BOS_TOKEN>` prepended — W0 verify. REUSE. | LOW arch (`Cohere2ForCausalLM`); accept the click-through gate at W0. Command-R-v01 35B (~70 GiB) deprioritized |
| **Phi-1 / Phi-2** | `microsoft/phi-2` | 2.78B | ~5.6 GiB | ungated | GPT-2 ByteLevel BPE (codegen); no BOS. REUSE. | LOW; lower recency |
| **MiniCPM3** | `openbmb/MiniCPM3-4B` | ~4B | ~8 GiB | ungated | Llama SentencePiece. REUSE. `trust_remote_code`. | MEDIUM — MLA config needs `trust_remote_code`; W0 probes oracle construction |
| **OLMo-3** (landed row W5) | `allenai/OLMo-3-1025-7B` | ~7.3B | ~13.6 GiB | cached | ByteLevel BPE (no BOS, OLMo-2 family). REUSE. | **BLOCKED** (RUN-VERIFIED W0 2026-07-26) — 0.25.0 CONSTRUCTS `Olmo3Config` (AutoConfig) but CANNOT RUN the model: `olmo2.py:143` `rope_parameters["rope_theta"]` → `KeyError` (transformers 5.13.1 nests `rope_parameters` per layer-type). Config-construct ≠ model-run; the OLMo-2-spike "CONSTRUCTS ⇒ NONE risk" claim conflated them. Reuses landed Gemma-3 sliding window |

### 0.4 Oracle-support: names that are pin-REMOVED (no SACRED bar available)

These appear in requests but are in vLLM's `_PREVIOUSLY_SUPPORTED_MODELS`
(`registry.py:701-712`) — the pinned 0.25.0 oracle CANNOT construct them, so there is
no SACRED gate vehicle; flag any of them for a W0 probe and treat as DEP-blocked until
the oracle advances:
- `Phi3SmallForCausalLM` (removed 0.9.2), `Phi4FlashForCausalLM` (0.10.2),
  `Phi4MultimodalForCausalLM` (0.12.0), `InternLM2VEForCausalLM` (0.23.0),
  `MotifForCausalLM` (0.10.2).
- There is **no bare `Phi4ForCausalLM` text row** in the pin — Phi-4 *dense* uses the
  `Phi3ForCausalLM` architecture (verified: `microsoft/phi-4` and `Phi-4-mini-instruct`
  both report `architectures: ["Phi3ForCausalLM"]`). The registered `Phi4*` names are
  all multimodal (`Phi4ForCausalLMV` siglip `:518`, `Phi4MMForCausalLM` `:519`) — out
  of scope (vision towers not started).

### 0.5 RANKED implementation queue (recency x GB10-fit x additivity)

**Tier 1 — cheapest zero-new-kernel, do first (one impl agent each, OLMo-2-style):**

| Rank | Family / arch | Row | Why here | One-line W-order |
|---|---|---|---|---|
| 1 | **OLMo-3** (`Olmo3ForCausalLM`) | (landed olmo2 row, W5) | IMPLEMENTED on the olmo2 class (dual rope + `layer_types` sliding window, reusing the landed Gemma-3 window); **but oracle-BLOCKED — NOT a reachable win.** RUN-VERIFIED W0 2026-07-26: 0.25.0 constructs `Olmo3Config` yet CANNOT run the model (`olmo2.py:143` `rope_parameters["rope_theta"]` → `KeyError`; transformers 5.13.1 nested rope schema). NO SACRED bar (D5) until an oracle advance | W5 code DONE; gate BLOCKED. Pivot the next impl agent to StableLM (Granite-3 already SACRED) — oracle-certain + ungated |
| 2 | **Phi-3 / Phi-4** (`Phi3ForCausalLM`) | `phi3-phi3-for-causal-lm` | Most mainstream recent dense; a `LlamaForCausalLM` subclass -> reuses the DONE Llama forward VERBATIM; ungated Phi-4-mini fits; oracle-certain | W0 config/BOS/tokenizer(SP vs BPE) -> W1 registry+config (su/longrope flag) -> W2 loader (packed qkv/gate_up, tied Phi-4 / untied Phi-3) -> W3 forward (reuse Llama block) -> W4 SACRED on Phi-4-mini + STRICT on phi-4-14B |
| 3 | **Granite-3** (`GraniteForCausalLM`) | `granite-granite-for-causal-lm` | Recent (IBM 2025); ZERO new kernel (4 default-1 scalar multipliers); ungated 2.5B fits | W0 -> W1 registry+config (4 multipliers as default-1 scalars) -> W2 loader -> W3 forward (Llama block + threaded multipliers) -> W4 SACRED on granite-3.3-2b |
| 4 | **StableLM** (`StableLmForCausalLM`) | `stablelm-stablelm-for-causal-lm` | ZERO new kernel (LayerNorm dense + partial rope + optional qkv bias, all REUSE); ungated 1.6B fits | W0 -> W1 registry+config (partial_rotary_factor, use_qkv_bias, LayerNorm eps) -> W2 loader -> W3 forward (LayerNorm block, reuse partial rope + kLayerNorm) -> W4 SACRED on stablelm-2-1_6b |
| 5 | **MiniCPM** (`MiniCPMForCausalLM`) | `minicpm-mini-cpmfor-causal-lm` | ZERO new kernel (scale_emb / scale_depth / dim_model_base scalars); dense checkpoint; MEDIUM oracle-risk (trust_remote_code + `.bin`) | W0 (confirm oracle constructs it + `.bin` loader path) -> W1 config scalars -> W2 loader -> W3 forward (Llama block + 3 scalars) -> W4 SACRED on MiniCPM-2B |

**Tier 2 — small-new-op, one small thing each:**

| Rank | Family / arch | Row | New op / delta | One-line W-order |
|---|---|---|---|---|
| 6 | **InternLM2** (`InternLM2ForCausalLM`) | `internlm2-intern-lm2-for-causal-lm` | fused-`wqkv` interleaved kv-group split (loader only) | W0 -> W1 config -> W2 loader (the `split_qkv` interleave -> merged qkv) -> W3 forward (Llama block) -> W4 SACRED on internlm2-chat-1_8b |
| 7 | **Command-R7B** (`Cohere2ForCausalLM`) | `commandr-cohere-for-causal-lm` | logit_scale scalar + parallel-residual block + per-head-LN qk-norm + interleaved sliding (REUSE) | W0 (BOS + LN mean-subtract + accept gate) -> W1 config -> W2 loader (tied, no bias) -> W3 parallel-residual block + qk-LN + logit_scale + sliding routing -> W4 SACRED on command-r7b |
| 8 | **Phi-1 / Phi-2** (`PhiForCausalLM`) | `phi-phi-for-causal-lm` | ONE new op: `kGelu`/NewGELU UNARY (non-gated); parallel-residual + biased qkv/lm_head + untied | W0 -> W1 `kGelu` unary op (unit-gated) -> W2 config+loader (biased qkv, untied lm_head bias) -> W3 parallel-residual LayerNorm block + partial rope + plain-GELU MLP -> W4 SACRED on phi-2 |
| 9 | **MiniCPM3** (`MiniCPM3ForCausalLM`) | `minicpm3-mini-cpm3-for-causal-lm` | MLA re-wire (reuses the landed DeepSeek-V2 MLA block) + MiniCPM scalars | W0 (oracle-construct probe) -> W1 config (q_lora/kv_lora dims + scalars) -> W2 loader (MLA fused_qkv_a + scalars) -> W3 forward (DeepSeek-V2 MLA block + MiniCPM scalar wiring) -> W4 SACRED on MiniCPM3-4B |

**Tier 3 — campaigns (sequence after Tier 1-2), each needs its own leaf spike:**
Falcon-H1 (`FalconH1ForCausalLM`, Mamba2 SSM hybrid — new state kernel),
GraniteMoeHybrid (SSM), GraniteMoe / GraniteMoeShared / Cohere2Moe / PhiMoE (FusedMoE
+ family-specific router/expert structure — reuse the BF16 grouped-GEMM but a distinct
row each), Falcon (`FalconForCausalLM`, older; alibi + `new_decoder_architecture`
parallel branches).

### 0.6 Recommended TOP 3 to implement next

1. **OLMo-3** (W5 on the landed OLMo-2 row) — nearly-free; W-order = W5 only.
2. **Phi-3 / Phi-4** (`Phi3ForCausalLM`) — W0->W1->W2->W3(reuse Llama)->W4 SACRED (Phi-4-mini + strict phi-4-14B).
3. **Granite-3** (`GraniteForCausalLM`) — W0->W1(4 scalars)->W2->W3(Llama block+multipliers)->W4 SACRED (granite-3.3-2b).

**ZERO-NEW-KERNEL near-additive count:** 4 of the 8 newly-scoped rows (Phi-3/Phi-4,
Granite-3, StableLM, MiniCPM); plus OLMo-3 (landed row W5) and the InternLM3/Yi Llama
aliases are also zero-new-kernel -> **7 zero-new-kernel bring-ups available**.

---

## 1. Structured contract

### Scope

Design — not build — a RANKED, grounded implementation queue for the next tier of
recent dense (and small-MoE) TEXT families, factoring each honestly into what REUSES
the landed substrate vs the specific new delta, with per-family GB10 fit, tokenizer,
and 0.25.0-oracle-support risk. Advances the 8 rows listed at the top
`INVENTORIED` -> `SPIKE`.

In scope: the triage table (§0.2) + hardware/tokenizer/oracle table (§0.3); the
pin-removed / no-bare-Phi4 oracle facts (§0.4); the ranked queue with per-row W-orders
and the TOP 3 (§0.5-0.6); the OLMo-3 (landed-row W5) and InternLM3/Yi (Llama-alias)
call-outs. Every reuse claim is anchored to a landed `src/`/`include/` op (§0.1).

OUT of scope, each with a reason: **implementation of anything** (spike — no code, no
kernels, no build, no benchmark, nothing downloaded; only HF API metadata read).
**MoE/SSM/older families** (Falcon, Falcon-H1, GraniteMoe*, Cohere2Moe, PhiMoE) — each
a distinct campaign row, stay `INVENTORIED` (§0.5 Tier 3). **Pin-removed names**
(Phi3Small, Phi4Flash, Phi4Multimodal, InternLM2VE, Motif) and **Phi-4 multimodal**
(`Phi4ForCausalLMV`, `Phi4MMForCausalLM`) — no constructible 0.25.0 oracle / vision
towers not started (§0.4). **The embedding/reward/MM rows** for these families
(`MODEL-EMBED-phi3-*`, `MODEL-REWARD-internlm2-*`, `MODEL-MM-minicpmv-*`,
`MODEL-MM-cohere*`, `MODEL-MM-granite*`, `MODEL-MM-phi*`) — separate modalities.

### Upstream chain

Registry (`vllm/model_executor/models/registry.py` @ `e24d1b24`): Phi-3 `:185`
(`phi3.py::Phi3ForCausalLM`, a `LlamaForCausalLM` subclass); Phi-1/2 `:184`
(`phi.py::PhiForCausalLM`); Cohere/Cohere2 `:84-85` (`commandr.py::CohereForCausalLM`);
Granite `:121` (`granite.py::GraniteForCausalLM`); StableLM/Epoch `:200-201`
(`stablelm.py::StablelmForCausalLM`); InternLM2 `:133` (`internlm2.py::InternLM2ForCausalLM`);
MiniCPM `:151` (`minicpm.py::MiniCPMForCausalLM`); MiniCPM3 `:152`
(`minicpm3.py::MiniCPM3ForCausalLM`); InternLM3 `:134` -> `llama.py::LlamaForCausalLM`
(alias). Removed: `registry.py:701-712`.

Model-layer anchors: `phi3.py:10-18` (subclass + packed map); `phi.py:77-320`
(parallel-residual `:201`, plain GELU `:166-168`, biased qkv/dense `:98,102`, untied
lm_head bias `:279,291`); `commandr.py:76-417` (Cohere LayerNorm `:76-87`, qk-norm
`:200-213`, sliding `:184-197`, parallel-residual `:264-272`, logit_scale `:376`, tie
`:372`); `granite.py:64-414` (multipliers `:137,240,245,313,371-372`, tie `:411-414`);
`stablelm.py:60-277` (partial rotary + `use_qkv_bias` `:124`, LayerNorm `:190-191,238`,
gate_up `:71-89`); `internlm2.py:53-327` (`wqkv`/`split_qkv` `:126-189`, `w1/w3->gate_up`
`:254-255`); `minicpm.py:82-576` (scale_emb `:443`, scale_depth `:384-393`, MoE class
`:82`, tie); `minicpm3.py:52-224` (MLA `:52-156`, subclasses MiniCPM `:186-224`).
Shared layers: `layernorm.py::RMSNorm`, `activation.py::SiluAndMul` /
`get_act_fn("gelu_new")`, `rotary_embedding/get_rope` (partial + su/longrope),
`attention/Attention` (`per_layer_sliding_window`, `logits_soft_cap` unused here),
`logits_processor.py::LogitsProcessor(scale=...)`.

### Our baseline

REUSED as-is (anchor + why) — see §0.1 for the full list: `vt::RmsNorm`
(`recipes.h`), `vt::LayerNorm` (`ops.h:181,1215`, landed for OPT), partial NeoX RoPE
(`ops.h:1243,1249,1310`, landed for GLM-4), `kSiluAndMul` (`ops.h:91`), `kMulScalar`
(`ops.h:201`, landed for Gemma), the DeepSeek-V2 MLA block (landed W6), the Gemma-3 /
OLMo-3 interleaved sliding-window + KV specs, the merged-column loader, tied/untied
embeddings, the `ByteLevel` + `LOAD-SENTENCEPIECE` tokenizer paths, the shape-agnostic
runner, `REGISTER_VLLM_MODEL`, the decode-graph sibling.

Honestly NOT reusable, and why (the per-family new delta): a plain **non-gated
NewGELU unary** for Phi-1/2 (we only have gated `kGeluAndMul`) — the ONE new compute op
in the batch; the **`wqkv` interleaved kv-group split** loader layout for InternLM2; the
**parallel-residual block** wiring + `logit_scale` scalar for Command-R; the MiniCPM3
**MLA re-wire** (reuses the landed block, new layer composition). Per the OPT/GLM/Gemma/
OLMo-2 precedent (their D2/D4), each family gets a NEW `<family>.{h,cpp}` block that
reuses only the glue + attention path — it does NOT extend `dense_attn_block.h::AttnBlock`
(which hard-codes pre-norm + Qwen per-head qk-norm). **No MODEL row is `DONE`** anywhere;
nothing here claims otherwise.

Precedent specs: [`sweep-olmo2.md`](sweep-olmo2.md), [`sweep-gemma.md`](sweep-gemma.md),
[`sweep-llama-3.2.md`](sweep-llama-3.2.md) (the su/longrope + Llama-subclass baseline),
[`sweep-opt-125m.md`](sweep-opt-125m.md) (LayerNorm + BOS trap),
[`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md) (MLA re-wire for MiniCPM3).

**Anchor-drift warning.** Re-anchor every cited `file:line` against the tree at
implementation time; anchors are against pin `e24d1b24` / base `5c00fc4`.

### Port map

| Upstream | Ours |
|---|---|
| `registry.py:185` `Phi3ForCausalLM` (`phi3.py`, Llama subclass) | **NEW (impl)** `phi3.{h,cpp}`/`_weights`/`_registry` reusing the Llama forward VERBATIM + optional su/longrope; ZERO new kernel |
| `registry.py:121` `GraniteForCausalLM` (`granite.py`) | **NEW (impl)** `granite.{h,cpp}` — Llama block + 4 default-1 scalar multipliers threaded (REUSE `kMulScalar`/folded scalars); ZERO new kernel |
| `registry.py:200-201` `StablelmForCausalLM` (`stablelm.py`) | **NEW (impl)** `stablelm.{h,cpp}` — LayerNorm dense (REUSE `kLayerNorm`) + partial rope (REUSE) + optional qkv bias; ZERO new kernel |
| `registry.py:151` `MiniCPMForCausalLM` (`minicpm.py`) | **NEW (impl)** `minicpm.{h,cpp}` — Llama block + scale_emb/scale_depth/dim_model_base scalars; ZERO new kernel |
| `registry.py:133` `InternLM2ForCausalLM` (`internlm2.py`) | **NEW (impl)** `internlm2.{h,cpp}` — Llama block; loader does the `wqkv` interleaved kv-group split -> merged qkv |
| `registry.py:84-85` `Cohere/Cohere2ForCausalLM` (`commandr.py`) | **NEW (impl)** `commandr.{h,cpp}` — parallel-residual block + Cohere LayerNorm (REUSE null-bias) + per-head qk-LN + `logit_scale` + interleaved sliding (REUSE Gemma-3) |
| `registry.py:184` `PhiForCausalLM` (`phi.py`) | **NEW (impl)** `phi.{h,cpp}` + ONE new op `kGelu` (NewGELU unary) — parallel-residual LayerNorm block + partial rope + biased qkv/untied-bias lm_head |
| `registry.py:152` `MiniCPM3ForCausalLM` (`minicpm3.py`) | **NEW (impl)** `minicpm3.{h,cpp}` — REWIRE the landed DeepSeek-V2 MLA block + MiniCPM scalars |
| `registry.py:134` `InternLM3ForCausalLM` -> `llama` | **REUSE** — a registry-alias string on the ACTIVE Llama row (not a new arch) |
| Falcon / Falcon-H1 / GraniteMoe* / Cohere2Moe / PhiMoE / removed names | **NOT PORTED** — out of scope (§0.4-0.5 Tier 3); stay `INVENTORIED` |

### Tests to port

Per [`.agents/porting.md`](../porting.md). Nothing is ported by THIS spike
(design only); this is the inventory that binds the implementing Ws.

| Upstream test | Tier | Ours (at impl time) |
|---|---|---|
| `tests/models/language/generation/test_common.py` (Phi-3, Granite, StableLM, InternLM2, MiniCPM, Command-R text-gen entries) | T-parity | `tests/parity/test_<family>_paged_engine.cpp` — the per-family SACRED token-exact gate (W4) |
| `tests/models/language/generation/test_granite.py` | T-parity | Granite text-gen correctness (W4) |
| `tests/models/registry.py` `_HfExamplesInfo` for each arch (+ pin-removed check) | T-unit | config/registry resolution + the removed-name loud-fail cases (W1) |
| `tests/models/test_initialization.py` (init smoke) | T-unit | init/registry resolution; MiniCPM/MiniCPM3 gate the trust_remote_code oracle-construct verdict (W0) |
| `tests/test_config.py` (arch-config resolution) | T-unit | arch-config resolution, gateable with NO checkpoint |
| `tests/v1/e2e/general/test_correctness_sliding_window.py` | T-e2e | Command-R7B (Cohere2) + OLMo-3 interleaved sliding window (W3/W5) |

### Gates (bind the implementing Ws; NONE run in this spike)

1. **Correctness (SACRED), per family** — token-exact vs the pinned vLLM 0.25.0
   oracle, greedy, identical prompts, gate form selected BY MEASUREMENT per
   [`near-tie-distributional-gate`](../verification.md) (run vLLM's own K=5 greedy first;
   distributional/near-tie-band fallback ONLY if measured, else STRICT; add a bigger
   STRICT model where the small one is a near-tie, e.g. phi-4-14B for Phi-3).
2. **New-op unit gate** — only Phi-1/2 has one: `kGelu`/NewGELU UNARY bit-exact vs the
   vLLM `gelu_new` reference at real dims. The other rows add no `vt::` op; their new
   facts (scalars, parallel-residual, wqkv split, MLA re-wire) are proven by gate 1
   (a wrong scalar/placement/split emits fluent-WRONG tokens — the OPT mode).
3. **Loader** — zero unmapped, zero missing on the gated checkpoint (incl. the
   InternLM2 `wqkv` interleave asserted, Granite/MiniCPM multipliers loaded, Command-R
   tied-no-bias, Phi biased-untied, MiniCPM3 MLA fused_qkv_a).
4. **Regression, non-negotiable** — all current SACRED gates UNCHANGED (each new family
   is new-files-only + at most one additive default-inert op, so existing gates are
   byte-identical by construction).
5. **Build** — clean full rebuild `-Werror`, zero warnings.
6. **memcheck** — `compute-sanitizer` zero errors on the new forward (+ `kGelu` for Phi).
7. **Records** — all five CI checkers green.
8. **SPEED** — explicitly PENDING and unclaimed; a row is `DONE` only at token-exact
   AND vLLM throughput on every axis.

### Dependencies

**No hard upward dependency** for the ZERO-NEW-KERNEL rows (Phi-3/4, Granite, StableLM,
MiniCPM) and OLMo-3 (W5 on landed infra). Phi-1/2 depends on the small `kGelu` unary.
MiniCPM3 depends on the landed MLA block (present). **Oracle preconditions (W0 probes):**
MiniCPM / MiniCPM3 need `trust_remote_code`; Command-R7B needs the HF click-through gate
accepted; OLMo-3 needs its checkpoint present (`Olmo3Config` construction already
verified). **Checkpoint downloads (not performed):** the §0.3 smallest genuine
checkpoints; stage sequentially ([[grid-per-sha-trees-fill-disk]]). **Downward
dependencies introduced:** the `kGelu` NewGELU unary (reusable by any non-gated-GELU
model, e.g. GPT-2/GPT-NeoX/Falcon later).

### Work breakdown (per-row W-orders are in §0.5; this spike delivers only the scoping)

- **This spike (DONE):** the triage + ranked queue + records; advance 8 rows
  `INVENTORIED` -> `SPIKE`; register `CLAIM-SWEEP-RECENT-DENSE`.
- **Next (separate impl claims, one agent each):** rank-1 OLMo-3 W5, rank-2 Phi-3/Phi-4,
  rank-3 Granite-3 — then StableLM, MiniCPM, InternLM2, Command-R7B, Phi-1/2, MiniCPM3,
  each per its §0.5 W-order and the gates above.

### Risks/decisions

**D1 — the queue is ordered by (recency x fit x additivity); OLMo-3 leads because it is
nearly-free on a LANDED row.** Phi-3/Granite lead the NEW rows because they reuse the
DONE Llama forward with zero (Phi-3) or scalar-only (Granite) deltas.

**D2 — do NOT extend `dense_attn_block.h::AttnBlock`.** Each family gets a new
`<family>.{h,cpp}` block reusing only the glue + attention path (the OPT/GLM/Gemma/OLMo-2
precedent). Command-R's parallel-residual and Phi's parallel-residual+plain-GELU are the
clearest cases that the shared header does not stretch.

**D3 — scalars/placement are silent-corruption hazards.** Granite's 4 multipliers,
MiniCPM's scale_emb/scale_depth/dim_model_base, Command-R's logit_scale, and the
parallel-residual re-join order all emit fluent-WRONG tokens if mis-wired (the OPT mode).
Mitigation: gate 1 is an e2e token-exact gate; each W3 grounds the exact scalar/residual
order line-by-line against the model file.

**D4 — BOS + tokenizer family is a per-row W0 verify.** SP vs ByteLevel varies within the
batch (even within Phi: Phi-3 SP, Phi-4 BPE). A silently mis-placed BOS scores 0/n while
emitting fluent text (OPT/Llama). W0 verifies BOS vs the oracle before any forward; the
gate can run tokenizer-free (feed the oracle's exact ids) if our loader does not validate
that family's tokenizer.json.

**D5 — oracle-support must be probed for MiniCPM/MiniCPM3 (trust_remote_code) and the
pin-removed names (§0.4).** A removed/unconstructible arch has NO SACRED bar; W0 records
the verdict and the row stays honestly blocked rather than claiming a gate it cannot run.

**D6 — MoE/SSM/older families stay `INVENTORIED` by design, not omission.** Falcon-H1 /
GraniteMoeHybrid are SSM campaigns; GraniteMoe / Cohere2Moe / PhiMoE add a family-specific
FusedMoE router/expert layout; Falcon is older with alibi + parallel `new_decoder_arch`
branches. Each is a distinct leaf spike (§0.5 Tier 3).
