# `Qwen4ExpForConditionalGeneration` (Qwen3.8-Flash-Next)

**Campaign row:** `MODEL-MM-QWEN4-EXP` (the ID carried by the branch, the issue and
the append-only index row)
**Model-matrix target row:** `MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation`,
the deterministic ID the row contract requires. Both name the same work; the index
row is append-only and cannot be re-keyed, so both are recorded rather than one
silently replaced.
**Issue:** [#1978](https://github.com/mudler/vllm.cpp/issues/1978)
**State:** `READY` (spec only; no product code lands under this row's first pull request)
**Motivating checkpoint:** `Qwen/Qwen3.8-Flash-Next`, released 2026-08-24, read live 2026-08-26

## Scope

Port `Qwen4ExpForConditionalGeneration` / `model_type: qwen4_exp`. The card calls it
"this experimental preview of the architecture that will underpin Qwen4"; the
`Qwen3.8` in the name is marketing continuity and not a shape relationship. It is a
180B-total / 6B-activated multimodal (image-text-to-text) hybrid: 48 layers in a
repeating `3 x linear_attention -> 1 x qwen_sparse_attention` pattern, 512-expert MoE
at top-10 plus one shared expert, a 20M-entry n-gram embedding table injected at
layer 2, a 4-branch gated residual stream, and a 1-layer MTP head.

In scope: text generation and the image/video path, every published quantized arm,
and the GGUF k-quant arms this repository requires of any model port.

Out of scope for the first implementation wave, each named under `## Owed` rather
than dropped: MTP depth > 1, the 1M-token RoPE extension the card advertises above
the native 262144, and any throughput claim.

### Merge sequencing for the `ACTIVE` transition and its claim (operator note)

W1 and W6a BOTH moved this row `READY -> ACTIVE` on their own branches, independently
and correctly — AGENTS.md "Records" requires the matrix row to move with the lifecycle
state, and each wave was the first product code from its own point of view. The result
is a collision that a clean three-way merge will NOT catch, and it is recorded here
because the second merge is where it bites:

- **The counts happen to be safe.** Both branches make the IDENTICAL edit, `ACTIVE`
  10 -> 11 and `READY` 4 -> 3, so a three-way merge with a base of 10/4 and both sides
  at 11/3 resolves to 11/3. That is luck, not design: two branches making DIFFERENT
  one-line edits to the same counter merge cleanly and apply BOTH, which is the failure
  AGENTS.md names under "Never store a measurement of one file inside another file".
  **Verify these two numbers by COUNTING ROWS at every merge, never by trusting the
  merge.**
- **The claim owner is NOT safe.** W1 wrote owner `CLAIM-MODEL-MM-QWEN4-EXP-W1` with
  `.agents/claims/CLAIM-MODEL-MM-QWEN4-EXP-W1.md`; W6a wrote `CLAIM-MODEL-MM-QWEN4-EXP`
  with its own file. Two different owners for one cell, and two claim files for one row.

**Resolution: the row-level claim `CLAIM-MODEL-MM-QWEN4-EXP` wins**, because the claim
covers the whole campaign rather than one wave, and `check-agent-record.py` binds an
owner to a ROW. Whichever of W1/W6a merges second drops its own transition and its own
claim file, keeping only the survivor. This is a merge-time reconciliation, not a
defect in either branch.

The same shape will recur for W2, W3 and W4: each is the first product code from its
own vantage, none of them should re-make the transition, and each should drop the edit
if it finds the row already `ACTIVE` on `main`.

## Why this needs a spec before code

Three of this row's decisions are expensive to reverse and cheap to get wrong, and
all three have already been made incorrectly once by an agent reading a related
record. They are settled here so a fresh implementer does not re-derive them.

1. **This is not a Qwen3.8 row.** `.agents/specs/qwen38-27b-bf16-gate.md` records
   `Qwen/Qwen3.8-27B` as the Qwen3.6-27B shape retrained, differing in exactly one
   config key. That precedent does not extend here. `qwen4_exp` shares an ancestor
   with `qwen3_5` and diverges in four load-bearing places.
2. **QSA's twin in vLLM is DeepSeek-V4's C4 indexer lane, not MiniMax-M3.** See
   `## Design`. This REVERSES the row's first reading, which rested on treating
   `MLAAttentionSpec` as an MLA claim; it is a per-state BUDGET shape, and M3 —
   itself plain GQA — uses it too. Nine independent structural matches tie QSA to
   DeepSeek-V4, `compress_ratio == 4` literally the same number. Building QSA on M3
   is the wrong port and it fails hard rather than subtly: M3 welds
   `SPARSE_BLOCK_SIZE = 128` to the KV page size, so ratio 4 forces a page size of 4
   and breaks `tl.dot`, whose tile needs >= 16. What M3 does contribute is a wiring
   precedent and not an algorithm: a plain-GQA model owning a key-only side cache
   through `MLAAttentionSpec`. The DSA/MLA reflex remains the trap, because this tree
   already has that path — the correction is which side of it QSA sits on
   ([#2049](https://github.com/mudler/vllm.cpp/issues/2049)).
3. **The oracle split is a direction, not a default.** See below.

## Oracles

**vLLM is this row's primary oracle from 2026-08-31.** `[Model] Support
Qwen3.8-Flash-Next (#53896)` landed at `e126687a9a828d513c01a07cd69f025f27d63280`
and added `vllm/models/qwen4_exp/`: a shared `common/`, a `config.py`, and two
complete backends, `nvidia/` and `amd/`. The registry carries three entries,
`Qwen4ExpForCausalLM` (`vllm/model_executor/models/registry.py:114`),
`Qwen4ExpForConditionalGeneration` (`:580`) and `Qwen4ExpMTP` (`:670`). Every
component this row lists except the vision tower now has a vLLM implementation
whose behaviour, defaults and naming this port must mirror.

**It is 595 commits ahead of the parity pin, and it is NOT reachable from the
pin.** `git merge-base --is-ancestor e126687a9a 555967922` exits 1, and so does the
reverse test. Every anchor cited from `e126687a9a` in this spec is therefore a
**forward reference to an unpinned upstream**, labelled as one at each use.
Advancing the pin belongs to the sync wave and NOT to this row; an implementer who
needs the ops at the pin still has none. Issue
[#2489](https://github.com/mudler/vllm.cpp/issues/2489) owns the reconciliation.

### The 2026-08-26 direction, and why it is superseded

**Recorded verbatim, and NOT deleted, because a record that quietly drops a
superseded reading cannot be audited.** The direction was: "use transformers as
oracle for algorithmic side. but use ops from vllm so we account for optimized
path."

The premise it rested on was stated in this section and was true when it was
written: **vLLM implemented nothing here.** Read live at `origin/main` =
`6a5e8f5979` on 2026-08-26, there was no `qwen4*` path, no registry entry, and a
repository-wide GitHub search for `qwen4` returned zero results. `vllm-omni`
likewise. On that premise the split was the correct reading of what each upstream
is, rather than a split of convenience: transformers
[#48337](https://github.com/huggingface/transformers/pull/48337) is a semantics
reference that says so in its own code, and its `Qwen4ExpTextQSAIndexer.forward`
loops in Python over `(batch_idx, query_idx)` under the comment "we only allow
eager and sdpa".

**The premise expired on 2026-08-31 and the direction expired with it.** AGENTS.md
makes vLLM the primary reference and "the only reference wherever it implements the
behavior", and requires a row to reconcile onto vLLM and record the change in the
spec once vLLM implements the path. The lane exception in
[`../oracles/transformers.md`](../oracles/transformers.md) also names its own
expiry in its own words: it "expires the moment vLLM registers `qwen4_exp`, at
which point the row reconciles onto vLLM and transformers demotes to the
preprocessing role it holds everywhere else."

**Nothing that was built under the direction is thereby wrong.** The reconciliation
below checks each component against vLLM rather than assuming either agreement or
divergence, and it found the two load-bearing algorithm calls the direction
produced, the unweighted mean pool and the divide-after-the-head-sum block score,
**confirmed** by vLLM's own kernels. What changes is the authority, not the answer,
and the components where the authority now differs are named individually.

### What each oracle is for, from 2026-09-01

| Oracle | Role now | Reach for it |
|---|---|---|
| `vllm` at `e126687a9a` (**unpinned**, forward reference) | **primary**: algorithm, ops, defaults, dtype policy, refusals, cache shapes, naming | every component of this model except the two below |
| `transformers` `v5.16.0` (lane pin) | **secondary**, demoted from algorithm to what it is everywhere else: the processor, tokenizer and checkpoint-semantics reference | preprocessing, the checkpoint's own tensor names and layout, and any question vLLM's two backends leave open |
| `llama-cpp-qwen4exp` (`ggml-org/llama.cpp` PR #27742) | unchanged | the GGUF conversion, its architecture string and its k-quant floor. vLLM reads safetensors and has no GGUF arm, so it cannot answer a GGUF question |
| `Qwen/Qwen3.8-Flash-Next` | unchanged | config and weights |

**The lane exception in `.agents/oracles/transformers.md` has fired its own expiry
condition and is spent.** It does not need to be argued down; it needs to be
recorded as expired, which is what this section does. The transformers pin itself
stays where it is, because `transformers` remains a registry oracle for its normal
role, and the lane pin is now the ordinary pin for that role on this row.

**One thing vLLM still does not supply, and it is the same thing as before.** vLLM
loads safetensors. Every GGUF question this row has, the quantized arms included,
is answered by `llama-cpp-qwen4exp` and by the checkpoint, and vLLM cannot rule on
it. That boundary is where the dtype question in #2477 actually lives, and the
reconciliation below says so precisely.

### Component-by-component reconciliation (Q4RECONCILE, 2026-09-01, #2489)

Read at vLLM `e126687a9a`, `vllm/models/qwen4_exp/nvidia/` unless a path says
otherwise. **Every vLLM anchor in this table is a forward reference to an
unpinned upstream.** `amd/` is the same shape and is not the mirror source for
this row; where the two could differ the NVIDIA path is authoritative.

Each divergence carries one of three verdicts. **(a)** this tree is wrong.
**(b)** a deliberate deviation that was argued against `transformers` or
`llama.cpp` and must now be re-argued against vLLM. **(c)** a structural
difference that does not change behaviour.

| # | Component | vLLM `e126687a9a` | This tree | Verdict |
|---|---|---|---|---|
| 1 | Model dtype | refuses every dtype but bf16, three times: `nvidia/qsa.py:188`, `nvidia/qsa.py:70`, `nvidia/indexer_qsa.py:113` | resolves bf16 from the GGUF (`qwen4_exp_gguf_weights.cpp:206`) | agree |
| 2 | RMSNorm weight dtype | the one model dtype. `GemmaRMSNorm.__init__` takes no dtype, so `nn.Parameter(torch.zeros(hidden_size))` gets `torch.get_default_dtype()`, which `model_loader/base_loader.py:53` pins to `model_config.dtype` for the whole construction; `default_weight_loader`'s `param.data.copy_(loaded_weight)` casts the checkpoint value into it | bf16 gammas: `LoadNormBf16` dequantizes and re-narrows (`qwen4_exp_weights.cpp:114-121`), used for all four QSA norms (`:409`, `:411`, `:423`, `:425`) | agree |
| 3 | RMSNorm activation dtype | bf16. `qwen3_next.py:424-431` chunks the gate off the **bf16** GEMM output and hands `q` to `self.q_norm` unwidened; both operands are promoted to f32 inside the kernel (`GemmaRMSNorm.forward_native`, `nvidia/ops/hc.py:42`, `:48`) | **f32 at one site.** `qwen4_exp_qsa_block.cpp:694` allocates `q_f32` as `DType::kF32`, and `:705` hands it to `vt::RmsNorm` beside a bf16 gamma | **(a)**, and it is #2477 |
| 4 | Grouped Gemma RMSNorm | `(1.0 + w.float())` over `group_size = hidden_size`, affine width `hc_count * hidden_size` (`common/hyperconnection.py:54-86`, `nvidia/model.py:263` and `:439` `hc_per_branch_norm=True`) | same polarity, same grouping, same `[10240]` width (`cuda_qwen4_exp.cu:290`, `ops.cpp:2536`) | agree |
| 5 | HC mix arithmetic | norm, down, `silu(x / hc_count)`, up, sigmoid, gated mean over the **normed** streams (`nvidia/hyperconnection.py:127-150`, `nvidia/ops/hc.py:100`, `:155`) | identical, including the division inside the SiLU and the mean over `normed` (`cpu_qwen4_exp.cpp:248-268`, `cuda_qwen4_exp.cu:302`, `:332`) | agree |
| 6 | HC down and inject GEMMs | **merged into one** `MergedColumnParallelLinear`, `[lora_rank, hc_count]` padded to a multiple of 16 rows, with a `WeightsMapper` that stacks the two checkpoint tensors (`nvidia/hyperconnection.py:98-110`, `nvidia/model.py:146-155`), and a decode plan keyed on that merged `(336, 10240)` shape (`nvidia/low_latency_gemm.py:72-79`) | three separate GEMMs, `hc_*_down` and `hc_*_inject` never stacked (`cuda_qwen4_exp.cu:414`, `:418`, `:428`) | **(b)** — upstream's merge is exactly the `vt::MergedGemmGroup` seam AGENTS.md mandates, and the padded 16-row shape is what its decode plan indexes |
| 7 | HC combine formula | `2 * sigmoid(inject(x_normed) / hc_count)`, inject read from the **normalized** hyper input (`common/hyperconnection.py:234-238`) | same, and also from `normed` (`cpu_qwen4_exp.cpp:282-286`, `cuda_qwen4_exp.cu:428`, `:347`) | agree |
| 8 | HC combine scheduling | **deferred** to the next mix boundary and **fused with that mix's RMSNorm** (`nvidia/hyperconnection.py:152-186` `combine_and_mix`, kernel `nvidia/ops/hc.py:267-331`); it is materialized early only where PLE adds into the stream (`nvidia/model.py:288-297`) | applied immediately, its own op, on the raw stream (`qwen4_exp_forward.cpp:466`, `:544`) | **(c)** for the arithmetic, **(b)** for the schedule: upstream reads the residual once instead of twice, and the fused kernel is the reason |
| 9 | HC projection quantization | never quantized. All three linears pass `quant_config=None` **explicitly** and `params_dtype=torch.bfloat16` **hardcoded**, not `model_config.dtype` (`nvidia/hyperconnection.py:102`, `:113`, `:122`; `nvidia/model.py:260`, `:436`); the decode GEMM then demands `weight.dtype == torch.bfloat16` and packed row-major on both operands (`nvidia/low_latency_gemm.py:95-96`) | `hc_*_down` / `hc_*_up` route through `GgufLoadPolicy` and may keep Q8_0 blocks (`qwen4_exp_weights.cpp:134-141`, admitted by name at `ops.cpp:2564-2568`) | **(b)** — see #2406 below |
| 10 | QSA block score | `sum_h relu(dot(k, q_h))` then `/ sqrt(index_head_dim)` **after** the head sum (`nvidia/ops/qsa.py:110-113`, divisor at `:628`) | the fold multiplies **before** the sum (`cpu_dsa_indexer.cpp:87`, `:103`, `:120`), with `n_head_scale = 1` and unit weights (`qwen4_exp_qsa_block.cpp:381-391`) | **(c)** — the reassociation W5b-4 already named, now confirmed against vLLM rather than transformers |
| 11 | Compressor pool | unweighted mean, `accumulator / COMPRESS_RATIO` (`nvidia/ops/qsa.py:543`) | unweighted mean, `acc / CR` (`cpu_qwen4_exp_qsa.cpp:152`, `cuda_qwen4_exp_qsa.cu:256`) | agree, and this **confirms** the call this spec made from transformers against the closest-named CuteDSL kernel |
| 12 | Compressor state | a per-request **ring of raw keys** sized `compress_ratio * cdiv(compress_ratio + num_speculative_tokens, compress_ratio)`, plus a **persistent compressed key cache**; a group spanning steps is closed out of the ring (`common/qsa_cache.py:763-783`, kernel `nvidia/ops/qsa.py:512-539`) | no ring. The raw prefix is cached in full and **every complete block is re-pooled on every step**, the pooled keys being per-call scratch (`qwen4_exp_qsa_block.cpp:287`, `:293`, `:305`) | **(b)** — a compute-for-memory trade upstream does not make, and it grows with context |
| 13 | Indexer side caches | **two per QSA layer**: raw `CircularBufferSpec(block_size=capacity, 1, head_size, bf16)` (`common/qsa_cache.py:775`) and compressed `MLAAttentionSpec(..., tokens_per_state=compress_ratio)` (`:789-794`). Both dtypes are `torch.bfloat16` **hardcoded** at construction (`nvidia/indexer_qsa.py:154`, `:163`) and re-checked in `bind_kv_cache` (`common/qsa_cache.py:725`) | **one**: `MLAAttentionSpec(..., compress_ratio=1)` at `ResolveKvCacheDType()` (`qwen4_exp_registry.cpp:1089-1099`) | **(b)** for the count, which follows from #12; **(b)** for the dtype, because upstream does not let the KV dtype knob reach this cache |
| 14 | With mRoPE, the raw side cache | widens to carry the exact three-axis int64 position beside each key, `rope_position_offset + 3*4` (`common/qsa_cache.py:733-752`) | no position tail; the pooled key is roped at the scalar `b * CR` (`cpu_qwen4_exp_qsa.cpp:183`) | **(c)** while the three mRoPE axes are forced equal (`qwen4_exp_forward.cpp:191-198`); **(a)** the moment the image or video path makes them differ |
| 15 | Pooled-key RoPE position | the **first member's** position, `end_position - COMPRESS_RATIO + 1`, read from the position cache when mRoPE is on (`nvidia/ops/qsa.py:548`, `:556-582`) | `b * CR`, which is the same value at a boundary (`cpu_qwen4_exp_qsa.cpp:183`) | agree |
| 16 | Top-k output shape | a fixed `[rows, token_topk + compress_ratio - 1]` int32 buffer, block ids expanded into it by `expand_qsa_block_indices_cuda` (`nvidia/ops/qsa.py:757`, `:800`), `block_topk = token_topk // compress_ratio` (`:766`) | no token buffer; block `b` becomes addresses `[b*CR, b*CR+CR)` inside the gather (`cpu_qwen4_exp_qsa.cpp:302`, `cuda_qwen4_exp_qsa.cu:463`). The fixed width exists only in the test-only host reference (`qwen4_exp_qsa.h:150`) | **(c)** |
| 17 | Q/K/V projection | one fused `qkv_proj`, the output gate carried inside the doubled q half (`qwen3_next.py:392`, `:424-428`) | separate `attn_q` (at `qdim*2`), `attn_k`, `attn_v` matmuls (`qwen4_exp_qsa_block.cpp:692`, `:733`, `:741`); the doubled q half and the per-head split agree | **(c)** for the split, but the fusion is a `vt::MergedGemmGroup` question like #6 |
| 18 | Output gate | `flat_output * torch.sigmoid(gate)` on the **bf16** gate, after attention and before `o_proj` (`nvidia/qsa.py:430-431`) | same place, but the gate is materialized f32 to keep the sigmoid argument unrounded (`qwen4_exp_qsa_block.cpp:695`, `:794`) | **(b)** — wider than upstream, and a token gate cannot see it |
| 19 | PLE n-gram embedding | **IMPLEMENTED.** splitmix64 multipliers seeded per PLE layer, xor-mix of the shifted id planes, per-head vocabulary at the nth prime after `ngram_vocab_size_base`, offsets accumulated (`nvidia/ple_layer.py:151-436`, the splitmix64 hash at `:159-167`, the per-layer multipliers at `:210-226` and the id mix at `:422-436`) | same construction (`qwen4_exp_ple.cpp:112-117`, `:183-197`, `:236-303`) | **the Port map row flips**: this is no longer a component with no vLLM op |
| 20 | PLE dilated depthwise conv | **IMPLEMENTED.** `nn.Conv1d(hc*H, hc*H, ple_conv_kernel_size, groups=hc*H, dilation=short_conv_dilation)` with `short_conv_dilation = ngram_size` and a short-conv state (`nvidia/ple_layer.py:589-599`, `nvidia/model.py:699`) | `vt::Qwen4ExpPleConv`, same kernel size and same `dilation = ngram_size` (`qwen4_exp_ple_block.cpp:271-272`, `:576-581`) | **the Port map row flips** |
| 21 | PLE grouped norms | three `Qwen4ExpPLEGroupedNorm` over `hc*H` with `group_size = hidden_size`, key/query/conv (`nvidia/ple_layer.py:580-588`, class at `:50`) | three `vt::RmsNormGroup(gemma=true, group=H)` (`qwen4_exp_ple_block.cpp:488-539`) | agree |
| 22 | MTP | **IMPLEMENTED.** `Qwen4ExpMTP` is registered (`registry.py:670`); the drafter reuses the backbone with PLE forced off, fuses via `residual_linear_shared`, and emits two hidden streams (`nvidia/mtp.py:1-14`, `:159-241`) | not implemented. `mtp_num_hidden_layers` is parsed (`qwen4_exp.cpp:549`) and never consumed; the registry states the head is deliberately unregistered (`qwen4_exp_registry.cpp:16-20`) | **owed**, and upstream now defines the target |
| 23 | Decode GEMM | `QWEN4_EXP_GEMM_PLANS` selects a CuteDSL skinny GEMM by `(N, K)` and token count, sm103 and bf16 only (`nvidia/low_latency_gemm.py:26-79`, `:155-156`) | none for this model; the shared `vt::MatmulBT` seam, deliberately, rather than a private GEMV (`cuda_qwen4_exp.cu:219`) | **(c)** today; the plan table is a lever once a device exists |
| 24 | Vision tower | `Qwen4ExpVisionConfig(Qwen3VLVisionConfig)`, reused unchanged (`config.py:23-26`) | the Qwen3.5-Moe tower, reused unchanged | agree, **not re-verified in this pass** |

### What this pass could NOT determine

Named rather than left silent, because a reconciliation that hides its gaps is
worse than one that is smaller.

- **Top-k tie and emission semantics.** Upstream calls
  `torch.ops._C.cooperative_topk` / `persistent_topk` (`nvidia/ops/qsa.py:794-803`),
  which are CUDA kernels outside the model directory. This tree's contract is
  all-select-below-k, ties to the LOWER index, ASCENDING emission
  (`cpu_dsa_indexer.cpp:145-176`). The two agree on `block_topk` and on the shape;
  whether they agree on ties was NOT read, and it cannot be settled from
  `vllm/models/qwen4_exp/` alone.
- **The recurrent KV group.** The W5c argument for ONE uniform `MambaSpec` was
  derived at the pin. `nvidia/model.py:748-800` now declares PLE and GDN state
  through two separate `get_*_mamba_state_*_from_config` classmethods. Whether
  that changes the one-group conclusion was NOT determined here and needs its own
  read.
- **The vision path.** Reused unchanged on both sides by inspection of the config
  classes only. No forward was compared.
- **`amd/` against `nvidia/`.** Not diffed. This row mirrors NVIDIA.
- **Anything measured.** Nothing in this table was run. `gateable = no` still
  holds, and a transcription of upstream is a reference, never a test.

### The two live questions vLLM now answers

**#2477, `vt: cuda rmsnorm: weight dtype must match x`. The mismatch is on the
ACTIVATION side, not the weight side, and there is exactly one site.**

vLLM's answer to "what dtype are the RMSNorm weight and the activation" is: **both
are the one model dtype, bf16**, and the kernel promotes both to f32 internally.
The weight is bf16 because `GemmaRMSNorm` declares no dtype and is built under
`set_default_torch_dtype(model_config.dtype)` (`model_loader/base_loader.py:53`),
and because `default_weight_loader` casts the checkpoint tensor into that
parameter. The activation is bf16 because `qwen3_next.py:424-431` chunks the gate
off the bf16 GEMM output and never widens `q`. vLLM has **no** f32 activation
entering a norm anywhere on this architecture's NVIDIA path, and it refuses the
model outright if `model_config.dtype` is not bf16.

In this tree all four QSA gammas are bf16 (`LoadNormBf16`,
`qwen4_exp_weights.cpp:409`, `:411`, `:423`, `:425`), and two of the three
`vt::RmsNorm` sites are bf16 on both operands: `qwen4_exp_qsa_block.cpp:632`
normalizes `q_index_raw`, allocated at `hidden.dtype` (`:592`), and `:736`
normalizes `k_raw`, also `hidden.dtype` (`:733`). **The third is not.**
`:694` allocates `DBuf q_f32(d, DType::kF32, ...)`, `:696` splits the query and the
output gate into it, and `:705` hands that f32 tensor to `vt::RmsNorm` beside the
bf16 `w.q_norm`. `RmsNormKernelCuda`'s `VT_CHECK(w.dtype == x.dtype)`
(`cuda_ops.cu:463`) then fires.

**`:705` is the UNCONDITIONAL one, and that is what makes it the site rather than
a candidate.** Its `x` is a literal `DType::kF32` allocation, so it mismatches a
bf16 gamma whatever the model dtype resolves to. The other two track
`hidden.dtype` and agree with the gamma whenever that is bf16, which the GGUF
path fixes at `qwen4_exp_gguf_weights.cpp:206`. It also sits in `QsaBlockCore`
(`qwen4_exp_qsa_block.cpp:417`) outside any `if (paged)`, so both arms reach it,
including the paged one that `qwen4_exp_forward.cpp:449` actually calls.

**One ordering caveat, stated because the issue asks for the order.** `:632` runs
BEFORE `:705` in every QSA layer. If `hidden.dtype` were ever not bf16, `:632`
would fire first and `:705` would never be reached. So `:705` is the site under
the dtype this tree resolves; it is not proof that no earlier site can fire under
a different one. A per-op trace, which #2477 already owes, settles that and
nothing here replaces it.

**Why it is CUDA-only.** `RmsNormKernel` on the CPU widens `w` and `x` to f32
independently and folds the gemma `+1` into the widened weight
(`src/vt/cpu/cpu_ops.cpp:546-566`). It accepts the mixed pair, which is why the
same model serves on `--device cpu` and refuses on `--device cuda`. The generic
dispatcher agrees with the CPU arm: `ops.cpp:1030` asks only `IsFloat` of each
operand. So the CUDA equality check is stricter than this tree's own contract as
well as than vLLM's.

**What the fix is not.** Widening the check is the one repair vLLM does not
support, because upstream's norm input is bf16 and an f32 one is a model-path
buffer that is too wide, which no token gate can see (AGENTS.md, "Inherit vLLM
defaults"). The f32 `q_f32` and the f32 `gate` exist to keep the sigmoid argument
unrounded, a deviation this spec records at row 18 and which now has to be argued
against `nvidia/qsa.py:425-427` rather than against transformers.

This is a finding, not a repair. The wave that owns #2477 owns the change.

**#2406, device-blind `quant_repack` on the hyper-connection weights.**

vLLM's HC projections are **never quantized and never anything but bf16**. All
three linears pass `quant_config=None` explicitly, and `params_dtype` is the
literal `torch.bfloat16` rather than `model_config.dtype`
(`nvidia/hyperconnection.py:102`, `:113`, `:122`; the two config sites are
`nvidia/model.py:260` and `:436`, and the MTP one is `nvidia/mtp.py:229`). The
layout it then expects on device is stated by the decode GEMM's own predicate:
packed row-major `[N, K]`, `x.dtype == torch.bfloat16` **and**
`weight.dtype == torch.bfloat16` (`nvidia/low_latency_gemm.py:95-96`), and the
plan it looks up is keyed on the merged `(336, 10240)` HC shape
(`:72-79`). The specialization is installed only over an
`UnquantizedLinearMethod` (`:164`), which `quant_config=None` guarantees.

**What that does and does not settle.** vLLM loads safetensors and never meets a
Q8_0 hyper-connection weight, so this is not a proof that a keep-quant HC arm is
wrong. It is a positive statement that upstream's HC path is designed around an
unquantized bf16 row-major operand, and it is the strongest statement available
about intent. The GGUF question itself stays with `llama-cpp-qwen4exp`, which is
the only oracle that reads this file at all, and the argument recorded at
`ops.cpp:2564-2583` stands on that oracle. What changes is that its **opening
premise is now false**: it says vLLM "has never registered `qwen4_exp`", and vLLM
has.

**The concrete lever this hands #2406.** Dequantizing `hc_*_down` and `hc_*_up` to
bf16 at load would mirror vLLM exactly, and it would remove all 194 of the
repack-hazard tensors from the issue's population in one step, because a bf16
weight never enters `OwnGgufQuantBlocks` and so never carries an i8mm marker. That
is a trade against memory and against llama.cpp's own choice to keep them typed,
so it is a decision for #2406's owner and not a conclusion of this pass.

### Gateability

`gateable = no` for the vLLM oracle on this row, and the reason is unchanged and is
memory rather than software: see `## Hardware`. An oracle must demonstrably build
**and run the model**, and no published artifact fits any fleet device. Registration
is not gateability. `e126687a9a` gives this row an authoritative **source** to mirror
today; it does not yet give it a denominator, and nothing in this section should be
read as claiming one.

## Upstream chain

| Source | Revision | Role |
|---|---|---|
| `huggingface/transformers` | `v5.16.0` (first release containing `qwen4_exp`, landed by `#48337` merged 2026-08-26) | **secondary since 2026-08-31**, demoted from algorithm to preprocessing and checkpoint semantics; `models/qwen4_exp/modular_qwen4_exp.py` is the authored delta, `modeling_qwen4_exp.py` the generated expansion |
| `vllm-project/vllm` | **`e126687a9a`** (`[Model] Support Qwen3.8-Flash-Next (#53896)`, 2026-08-31; **NOT reachable from the parity pin `555967922`, and 595 commits ahead of it**, so every anchor into it is a forward reference to an unpinned upstream) | **primary**: algorithm, ops, defaults, dtype policy, refusals, cache shapes and naming, at `vllm/models/qwen4_exp/` with `nvidia/` and `amd/` backends |
| `Qwen/Qwen3.8-Flash-Next` | HF `main`, read 2026-08-26 | config and weights |

Read the **modular** file, not the generated one. It is 1186 lines against 2707 and
it is the file that states what is inherited unchanged, which is most of the model.

On the vLLM side read `nvidia/`, not `amd/`. The two are the same shape and this
row's target is CUDA; where they differ the NVIDIA path is the one to mirror, and
any component where they diverge is called out in the reconciliation above.

## Our baseline

What this tree already has, and therefore what the port does NOT rebuild. Stated
first because the delta only means something against it, and because the size of
this list is the reason the row is tractable at all.

- **The Qwen3.5 family end to end.** `src/vllm/model_executor/models/qwen3_5*.cpp`
  carries the dense and MoE backbones, the GGUF weights path, the MTP draft
  (`Qwen3_5MTPModel`) and the runner integration. `Qwen4ExpTextModel` inherits from
  `Qwen3_5MoeTextModel`, so this is the base the upstream delta is written against.
- **GDN linear attention with a Triton-AOT fast path.** `src/vt/cuda/cuda_gdn.cu`.
  The AOT specializations are pinned to `K=V=128, Hg=16, H in {48,32}` and this
  model's `linear_key_head_dim` / `linear_value_head_dim` / `linear_num_key_heads` /
  `linear_num_value_heads` are `128 / 128 / 16 / 48`. An exact hit, not a near miss.
- **A working sparse-attention indexer**, `deepseek_v4_dsa.cpp` +
  `deepseek_v4_compressor.h` + `src/vt/cuda/cuda_deepseek_v4.cu`. Useful for its
  compressor and its cache plumbing; **not** the right base for QSA's selection
  path, see `## Port map`.
- **Hyper-connection residual streams**, `deepseek_v4_mhc.cpp`, ported 1:1 from
  vLLM's `kernels/mhc/`. Different math from Gated Residual, same fused shape.
- **Interleaved mRoPE**, `layers/rotary_embedding/mrope.cpp`, which this model needs
  (`mrope_section [11, 11, 10]`, `partial_rotary_factor` 0.25 over `head_dim` 256).
- **The Qwen3.5-Moe vision tower**, which upstream reuses here **unchanged**.
- MoE with grouped GEMM, and the GGUF k-quant loader stack.

## Design

`Qwen4ExpTextModel` inherits from `Qwen3_5MoeTextModel` and leaves the rotary
embedding, MLP, experts, TopK router and the **entire vision tower** unchanged
(`class Qwen4ExpVisionModel(Qwen3_5MoeVisionModel): pass`). This tree already has all
of that. The port is the delta below.

## Port map

| Component | Algorithm oracle | Op oracle (vLLM) | This tree |
|---|---|---|---|
| GDN linear attention | `Qwen4ExpTextGatedDeltaNet` | `layers/mamba/gdn/qwen_gdn_linear_attn.py` | **HAVE.** `K=V=128, Hg=16, Hv=48` is an exact match for the AOT gate in `TryTritonPackedDecode` / the delta_h dispatch (`src/vt/cuda/cuda_gdn.cu`, pinned to `K=V=128, Hg=16, H in {48,32}`) |
| Grouped RMSNorm | `Qwen4ExpTextRMSNorm(group_size=)` | `layers/layernorm.py` **`RMSNormGated`** (`group_size`), NOT the plain `RMSNorm` | new, small; see the correction below |
| QSA block scoring + top-k | `Qwen4ExpTextQSAIndexer` | **DeepSeek-V4 C4 indexer lane**: `fp8_mqa_logits` / `top_k_per_row`, `v1/attention/backends/mla/indexer.py`. NOT MiniMax-M3, see below | new |
| QSA pooled-key build | indexer forward | the **Triton** `head_dim=128` compress/norm/RoPE/store kernel with `OVERLAP=False` and the pool replaced by a mean. NOT the CuteDSL `SparseAttnCompressNormRopeStoreC4Kernel`, which refuses `overlap=False` and pools by learned softmax over 8 | partial: `deepseek_v4_compressor.h` |
| Indexer side cache | `Cache.update_indexer` | `MLAAttentionSpec(num_kv_heads=1, head_size=128, tokens_per_state=4)` + `get_compressed_slot_mapping`, as-is; M3 supplies only the registration precedent | new KV spec |
| Gated Residual | `Qwen4ExpTextGatedResidual` | `layers/mhc.py`, `kernels/mhc/*` (**different math**, same fused shape) | partial: `deepseek_v4_mhc.cpp` |
| MoE 512 / top-10 + 1 shared / intermediate 640 | `Qwen4ExpTextSparseMoeBlock` | FusedMoE, grouped GEMM | HAVE, shape change only |
| MTP, 1 layer, `hybrid: true` | config `mtp` | **`nvidia/mtp.py`**, `Qwen4ExpMTP` registered at `registry.py:670`, since 2026-08-31; `qwen3_5_mtp.py` is no longer the nearest form | NOT implemented for this model, `## Owed` |
| PLE dilated depthwise conv | `Qwen4ExpTextPLELayer._short_conv` | **`nvidia/ple_layer.py:592-601`** since 2026-08-31: `nn.Conv1d(groups=hc*H, dilation=ngram_size)` plus a short-conv state. Was `NONE` | new when written; mirror the vLLM form now |
| N-gram hashed embedding | `Qwen4ExpTextNGramEmbedding` | **`nvidia/ple_layer.py:240-436`** since 2026-08-31: splitmix64 multipliers, xor mix, per-head prime vocabulary. Was `NONE` | new when written; mirror the vLLM form now |
| Vision tower | `Qwen4ExpVisionModel` = `Qwen3_5MoeVisionModel` | qwen3_5 vision | HAVE. `deepstack_visual_indexes: []`, so no deepstack |

**That was true until 2026-08-31 and it is not true now.** The sentence this
paragraph replaces read "exactly two components have no vLLM op", and the two it
named were the PLE conv and the n-gram embedding. `e126687a9a` implements both.
**No component of this model except the vision tower is now without a vLLM form**,
so nothing here is authored from transformers alone any more, and the two rows the
change touches are marked above. See `### Component-by-component reconciliation`
for what each one does and where this tree agrees with it. Mirroring the vLLM form
is mandatory rather than optional, and that obligation now reaches every row of
this table.

### QSA maps to DeepSeek-V4's C4 indexer lane, NOT to MiniMax-M3

**This reverses the call this spec was first written with, and the reversal is the
most important thing in the document.** The original reading was that QSA, being plain
GQA rather than MLA, had to map onto vLLM's non-MLA block-sparse case (MiniMax-M3) and
not onto DeepSeek's DSA. That reasoning was wrong, and it was wrong for a specific,
checkable reason recorded here so it is not repeated: **`MLAAttentionSpec` is not an
MLA claim.** M3's own indexer cache uses it while being a plain-GQA model, and the
comment beside it says why -- "Key-only: MLAAttentionSpec budgets one vector/token (not
2x for K+V)". It is a budget shape, not an architecture assertion. Once that prop is
removed, the GQA-versus-MLA argument for preferring M3 collapses entirely.

Verified at vLLM `origin/main` = `6a5e8f5979`.

**Nine independent structural matches with DeepSeek-V4**, and `compress_ratio == 4` is
literally the same number:

| QSA (transformers v5.16.0) | DeepSeek-V4 (vLLM) |
|---|---|
| index MQA, 1 key head, dim 128 | index MQA, 1 key head, dim 128 |
| score `relu(q.k)` summed over index heads | `(score.relu() * weights).sum(dim=0)` |
| scale `1/sqrt(indexer_head_dim)` | `softmax_scale = head_dim ** -0.5` |
| one score set per query token, no head axis | `topk_indices_buffer[num_tokens, topk]` |
| pool `compress_ratio` tokens into one key | boundary `(position + 1) % COMPRESS_RATIO == 0` |
| `k_layernorm` on the pooled key | RMSNorm on the compressed key |
| RoPE at the **block-start** position | `compressed_pos = (position // CR) * CR` |
| candidates = `visible // compress_ratio` | `len_per_token = (start_pos + 1 + offset) // CR` |
| one stored state per 4 tokens | `MLAAttentionSpec(tokens_per_state=compress_ratio)` |

`tokens_per_state` is a first-class KV-cache field upstream, documented as "Ints > 1
compress multiple tokens into one state (DSv4 sparse MLA)". It is exactly what QSA's
side cache needs, and it does not exist on the M3 path.

**Why M3 is not merely a worse fit but a different algorithm.** Its score is
`tl.max(qk, axis=1)` over 128 **raw** token dots -- no pooling stage, no relu, no head
reduction -- and it asserts `num_idx_heads == num_kv_heads` with the comment "no topk
index reduce", so it produces one independent block set **per KV head** where QSA
produces one set per token. Its `SPARSE_BLOCK_SIZE = 128` is not a tunable: the file
states "One sparse block == one KV page", and both the score and the attend index
`block_table[blk]` on that identity. Moving it to 4 would force a KV page size of 4 and
break `tl.dot`, whose tile needs at least 16.

**M3 still contributes exactly one thing, and it is a wiring precedent rather than an
algorithm:** the demonstration that a plain-GQA model can own a key-only side cache
through `MLAAttentionSpec` and a private indexer backend registered into
`static_forward_context`. Take that pattern; take no kernel.

**The genuinely new work is the CONSUMER, and nothing upstream supplies it.** Every
DSv4 sparse consumer attends to the **compressed** MLA KV, one state per four tokens.
M3's consumers attend to raw tokens but only at page granularity. QSA attends to **raw
tokens selected at ratio-4 granularity**, which no vLLM consumer does. The port has to
expand block id `b` into tokens `[4b, 4b+4)`, append the ragged tail, and run dense GQA
(24 query heads over 2 KV heads, `head_dim` 256) across the gathered set.

**Two silent-failure traps follow, and both would pass a naive gate.**

1. Wiring QSA's top-k straight into a DSv4 sparse-MLA consumer attends to a **pooled**
   key and value instead of the four real tokens. It still produces plausible output.
   A short-prompt token gate cannot catch it, because at context <= `indexer_budget`
   every candidate is selected and the only remaining difference is the value pooling.
   Any QSA gate must therefore run past 2048 tokens of context to be worth anything --
   a requirement this spec did not previously state and which changes what `## Gates`
   has to demand.
2. `SparseAttnCompressNormRopeStoreC4Kernel` does **not** mean-pool, despite being the
   closest-named kernel. It is a learned softmax-weighted pool over an **overlapping
   window of 8** driven by a score channel QSA's checkpoint does not have, and the
   CuteDSL variant refuses `overlap=False` at compile time. Its scaffolding is a direct
   match -- boundary predicate, block-start RoPE, paged store -- but the pooling
   operator must be replaced with an unweighted mean over a non-overlapping window of
   4. The **Triton** `head_dim=128` variant, where `OVERLAP` is a plain `constexpr`, is
   the correct starting point; the CuteDSL C4 one is not.

Also reconcile, and do not inherit: DSv4 has a `weights_proj` producing per-head logit
weights that QSA has no tensor for (QSA's weight is the constant `1/sqrt(128)`), and
its RoPE is GPT-J-style over a trailing contiguous span, whereas QSA uses interleaved
mRoPE over the **leading** 64 dims with the NoPE dims trailing -- the halves are
swapped end for end.

### W5b-4 correction: the indexer's SCORE and TOP-K are ALREADY `vt::` ops

Issue [#2167](https://github.com/mudler/vllm.cpp/issues/2167) opened with a
"why nothing existing serves it" table naming `IndexSelect`, `TopKValuesIndices`,
`GatherMlaCache` and the fused `kDeepseekV4Dsa` / `kDeepseekV4Compressor`. **That
table is incomplete, and the two ops it omits are the two that do serve.** The
correction is recorded here rather than in the issue, which is append-only in
practice, because a later wave reading the issue would otherwise re-derive it.

`vt::DsaIndexerLogits` (`include/vt/ops.h`, kernel `src/vt/cpu/cpu_dsa_indexer.cpp`)
computes, over a ONE-key-head MQA cache with a per-query `[win_start, win_end)`
window:

```
logit[t,s] = sum_h fold[t,h] * ReLU(dot(q[t,h,:], k[s,:]))
fold[t,h]  = weights[t,h] * q_scale[t,h] * softmax_scale * n_head_scale
```

With `weights` all ones, `q_scale` null and `n_head_scale = 1`, the fold
collapses to the single constant `softmax_scale`. Set that to
`index_head_dim ** -0.5` and this **is** `Qwen4ExpTextQSAIndexer`'s block score
(`modeling_qwen4_exp.py:690-693`): QSA has neither DeepSeek-V4's learned
`weights_proj` nor its `n_head ** -0.5`, so the constant is all that is left.
`vt::DsaTopkSelect` is the same all-select-below-k, ties-to-the-LOWER-index,
ASCENDING-emission top-k — the exact three semantics `QsaTopkBlocks` inherited
from `sampler.cu` in W4 — applied to the block axis instead of the token axis.

So the QSA indexer is `Qwen4ExpQsaCompress` followed by those two, with
`win_start[t] = 0` and `win_end[t] = kv_len[t] / compress_ratio`. Adding a
QSA-private scoring kernel beside `cpu_dsa_indexer.cpp` would have been the
parallel path AGENTS.md §"Shared seams" forbids, in the same file that already
declines to re-implement `k_norm` and the leading-slice rope for precisely that
reason.

**One reassociation survives, and it is named rather than hidden.** Upstream
divides AFTER the head sum and the fold multiplies BEFORE it — `c * sum_h r_h`
against `sum_h c * r_h`. Equal in exact arithmetic, up to an ulp apart in f32,
and top-k is invariant under a positive scalar, so no selection can move except
through a tie manufactured at that ulp. That is an argument, not a measurement,
which is why `tests/vllm/models/test_qwen4_exp_qsa_device.cpp` compares the
COMPOSED selection against the lane-pinned oracle's own selected token sets for
every query token of both fixtures, ragged tail included, rather than relying on
it. Mutation M27 is the paired control: making `weights` non-uniform breaks the
collapse and reds the suite, so the ones are load-bearing rather than decorative.
M26 is the other half — inheriting DeepSeek-V4's `n_head_scale` SURVIVES, which
is the same positive-rescale blindness W4 already recorded for `QsaBlockScores`,
and it is in the table so the pair reads as an instrument that is wired up.

**What remains genuinely new is two ops.** The mean pool has no `vt::`
counterpart at all — this tree has no mean, no pool, no axis reduction and no
transpose, so the non-overlapping window cannot even be faked as a strided
depthwise conv — and fusing it with the norm and the block-start rope mirrors
upstream, whose compressor is one kernel, on the in-tree `kFusedNormRope`
precedent. The GATHER consumer has no counterpart anywhere: every DeepSeek-V4
sparse consumer attends the COMPRESSED MLA KV and MiniMax-M3's attend raw tokens
at KV-PAGE granularity, while QSA attends RAW tokens selected at ratio-4
granularity.

### Two structural consequences beyond the module list

- **The residual stream is `hc_count * hidden_size` = 4 x 2560 = 10240 wide through
  the whole stack.** `Qwen4ExpTextGatedResidual` reads it through a grouped RMSNorm
  and a low-rank (`hc_lowrank` = 320) SiLU-then-sigmoid gate, collapses to 2560 for
  the block, and writes back with a per-branch scalar gate
  (`2 * sigmoid(block_inject_weight(x) / hc_count)`). This is a change to the
  per-layer loop and to every residual buffer, not a drop-in module. The
  `Qwen4ExpTextModel` also holds one `use_combine=False` mixer that collapses the
  stream at the end.
- **`number_of_conv_states = 3` on a PLE layer** (GDN conv, PLE conv, and the n-gram
  token history, which upstream stores as a third conv state precisely because the
  manipulations are identical). The KV-cache spec grows a third conv stream plus the
  indexer side cache. Adjacent to [#1963](https://github.com/mudler/vllm.cpp/issues/1963)
  and [#1966](https://github.com/mudler/vllm.cpp/issues/1966).

### The KV-cache spec is THREE groups and ONE uniform recurrent group (W5c, #2031)

`MakeQwen4ExpKVCache` returns instead of refusing. Landed by W5c; the shape and
the reason are recorded here because the alternative shape is the one a reader
arrives with.

| # | layer_names | spec |
|---|---|---|
| 0 | the 12 QSA layers, `model.layers.<l>.self_attn.attn` | `FullAttentionSpec(block, 2, 256, ResolveKvCacheDType())` |
| 1 | the 36 linear layers, `model.layers.<l>.linear_attn` | `MambaSpec(block, {{10240,3},{48,128,128},{10240,9},{2}}, {bf16, ssm, bf16, kI64})` |
| 2 | the 12 QSA layers, `model.layers.<l>.self_attn.indexer.k_cache` | `MLAAttentionSpec(block, 128, ResolveKvCacheDType(), 1, …, compress_ratio=4)` |

**ONE uniform recurrent group, not per-layer specs and not several groups, and
that is the MIRROR rather than a shortcut.** Only ONE of the 36 linear layers
carries the PLE conv and the n-gram history, so a per-layer spec set is the
shape a reader expects. Upstream cannot produce it. Read at the pin
`5559679229`:

- `vllm/model_executor/models/interfaces.py:809-812` —
  `get_mamba_state_shape_from_config(cls, vllm_config)` is a CLASSMETHOD over
  the CONFIG, with no `layer_idx`. 19 definitions of that name tree-wide: this
  protocol declaration plus **18 implementations**, and not one of them takes a
  layer index.
- `vllm/v1/worker/mamba_utils.py:441` — `get_mamba_groups` asserts
  `all(mamba_specs[0] == spec for spec in mamba_specs)`: every `MambaSpec` in
  the model EQUAL, `shapes` and `dtypes` included.
- `vllm/v1/core/kv_cache_utils.py:1101-1109` — a `MambaSpec` whose page is
  smaller than the model's max is given `page_size_padded=max_page_size` and is
  otherwise unchanged. Upstream PADS. It does not split.

**The cost, derived and not measured.** The PLE conv is `10240 x 9` at bf16 =
184320 B and the n-gram history is 2 `int64` = 16 B, so **184336 B per sequence**
on each of the **35** linear layers that never read them: **49.2 MiB** at the
default `max_num_seqs` of 8, 0.09% of the GB10 headroom `## Hardware` accounts.
Gated as a literal in `tests/vllm/models/test_qwen4_exp_kv_cache.cpp` against
the same config with `ple_layer_ids` erased, so the number moves if the shapes
do. **No device has allocated it** — see `## Owed`.

This CORRECTS `.agents/specs/recurrent-multistate.md`, whose `## Owed` said a
second recurrent group "IS on a PLE topology's path". That measurement fed
upstream's grouping functions a heterogeneous per-layer input upstream never
constructs. Both halves stay owed as generic engine debt —
`ComputeHybridKvBudget` reads only the first mamba group
(`src/vllm/v1/core/hybrid_kv_budget.cpp:26`) — and neither is on this row's path.

**State order is `[gdn_conv, temporal, ple_conv, ngram]`, a deliberate
divergence from upstream's list order, and the same bytes.** Upstream keeps the
three CONV states adjacent (`number_of_conv_states = 3`) with the temporal state
after them. `GdnStateCache` publishes `conv_state = states[0]` and
`ssm_state = states[1]` as NAMED fields that THREE model families read —
`qwen3_5.cpp`, `kimi_linear_device.cpp`, and the `nemotron_h` pair
`nemotron_h_device.cpp` / `nemotron_h_forward.h` — so moving the temporal state
off slot 1 would silently re-point three model families. Slice order differs;
`page_size_bytes` does not.

**Three, not four ([#2203](https://github.com/mudler/vllm.cpp/issues/2203)).**
This wave first wrote FOUR here and in `qwen4_exp_registry.cpp`, inheriting the
list from `.agents/specs/recurrent-multistate.md` (landed by `f7710c1b4`,
[#2131](https://github.com/mudler/vllm.cpp/issues/2131)), whose fourth name is
`gemma4_mm.cpp`. That file reads NEITHER field — zero occurrences of
`conv_state` and zero of `ssm_state` — and its only two mentions of the type are
an include comment and `std::vector<GdnStateCache> no_gdn_state;`
(`gemma4_mm.cpp:221`), passed EMPTY. It is the file that proves Gemma-4 has no
recurrent arm, cited as proving the opposite. `muse_glimmer_mm.cpp:340` and
`qwen3_vl.cpp:621` carry the identical empty-vector shape, so the wrong fourth
name was one of the three files that demonstrate the negative. Measured on
`ad6696fa3`, `GdnStateCache` / `conv_state` / `ssm_state` counts per file:

| File | `GdnStateCache` | `conv_state` | `ssm_state` |
|---|---|---|---|
| `qwen3_5.cpp` | 37 | 33 | 34 |
| `nemotron_h_device.cpp` | 6 | 9 | 14 |
| `kimi_linear_device.cpp` | 2 | 7 | 6 |
| `gemma4_mm.cpp` | 2 | **0** | **0** |
| `muse_glimmer_mm.cpp` | 2 | **0** | **0** |
| `qwen3_vl.cpp` | 2 | **0** | **0** |

A grep on the FIELD name over-counts in the other direction:
`glm5_next_kda.cpp:343-345` matches `conv_state` 13 times, but that is
`Glm5NextKdaCache::conv_state`, a `std::vector<float>` KDA sequence state
(`include/vllm/model_executor/models/glm5_next_kda.h:314`), where this one is a
`vt::Tensor` (`include/vllm/model_executor/models/qwen3_5.h:111`); that file has
zero occurrences of `GdnStateCache`. Grep the TYPE. **The conclusion does not
move:** re-pointing three families is still the reason the temporal state stays
on slot 1. Only the enumeration was wrong, and no checker can see this class —
`check-symbol-anchors` resolves symbols, and `GdnStateCache` genuinely appears
in `gemma4_mm.cpp`, so symbol existence passes on a file whose behaviour is the
opposite of the one asserted. Same class as
[#2198](https://github.com/mudler/vllm.cpp/issues/2198), which this wave closes.

**Group 2 must be an `MLAAttentionSpec`, and a `FullAttentionSpec` there fails
SILENTLY.** The runner's leftover scan treats the first published
`kFullAttention` group that is neither the target nor the recurrent one as the
single `fa_draft` draft-KV slot and `continue`s
(`src/vllm/v1/worker/gpu/runner.cpp`, the `draft_slot_taken` arm). The leftover
count then stays 0, `multi_cache_topology` stays false, the legacy one-buffer-
per-layer path runs, and the side cache is published and never allocated with
nothing reported. `kMlaAttention` is not absorbed by that arm, so the topology
is multi-cache and every published cache gets a buffer.

**Real per-layer names, never placeholders.**
`ResolveKVCacheGroupLayerNames` rewrites a placeholder group set, but its
fallback can name only a TARGET attention group and one `fa_draft` slot: a THIRD
attention group reaches `group.layer_names.clear()`
(`src/vllm/v1/kv_cache_interface.cpp`), and an unnamed group is then refused by
the runner's multi-cache admission check for names that "do not all resolve to
distinct in-range layer indices". Publishing real names also makes that rewrite
a no-op by its own idempotence guard, gated in the KV suite.

**`block_size % compress_ratio != 0` is refused BY NAME**, because
`storage_block_size()` is integer division
(`vllm/v1/kv_cache_interface.py:393-395`) and truncating it sizes the page for
fewer states than the block covers — a short cache, i.e. wrong tokens rather
than a crash. Upstream never meets it (its DeepSeek-V4 block sizes are powers of
two above the ratio); ours arrives as a caller-supplied parameter.

**A non-uniform `compress_ratio` was ALREADY refused and was NOT gated.**
`Qwen4ExpHfConfigFromGguf` takes the first non-zero entry of the per-layer
`attention.compress_ratios` and requires the rest to agree — the mirror of
upstream's `MLAAttentionSpec.merge` assert
(`vllm/v1/kv_cache_interface.py:424-435`) — and deleting that `VT_CHECK` left
`test_qwen4_exp_gguf_weights` fully green. W5c gates it rather than adding a
second copy in the KV builder. The fixture has to DOUBLE `block_count` to reach
it: at four layers and `full_attention_interval` 4 there is exactly one sparse
layer, one non-zero ratio cannot disagree with itself, and a stray non-zero on a
linear layer is caught one check earlier by the schedule agreement.

**Two refusals W5c did NOT add, because W1 already has them**, verified rather
than assumed: a `ple_layer_ids` entry outside the one-indexed range, and a PLE
id landing on a layer the rewrite made sparse
(`src/vllm/model_executor/models/qwen4_exp.cpp`), both gated by named subcases
in `test_qwen4_exp_scaffold.cpp`. A second copy in the KV builder would be a
second derivation of one rule.

### The n-gram embedding is integer-exact or it is silently wrong

Derived from the lane pin and then **verified against the published checkpoint** by
range-reading the safetensors payload, so these are read values and not predictions.

`config.seed` is absent from the published config, so the dataclass default **1234**
applies. That was confirmed rather than assumed: reconstructing the splitmix64 chain
at seed 1234 gives `layer_multipliers = [23703573157769, 20109073645365,
8052911324071]`, and a range read of that buffer out of
`model-00005-of-00131.safetensors` returns those three values exactly.

Head vocab sizes are the successive primes after `ngram_vocab_size_base - 1`, so head
0 is 20000003 and head 15 is 20000171; `total_vocab_size = 320001446`, padded to
**320001536** (90 unaddressable rows), giving `320001536 x 160 = 51,200,245,760`
parameters. `ngram_heads_vocab_sizes` and `ngram_heads_offsets` were range-read from
the checkpoint and match the derivation entry for entry.

**The three C++ divergence sites, ranked.** All are silent.

1. **`_splitmix64` must be `uint64_t` throughout.** Its `>> 30 / 27 / 31` are logical
   shifts on a non-negative Python int; on a signed `int64_t` they become arithmetic
   shifts and the multiplier is wrong. The value has its top bit set about half the
   time, so this fires immediately.
2. **`_splitmix64(value) % half_bound` must be an unsigned modulo.** The dividend
   routinely exceeds 2^63; a signed modulo yields a negative residue.
3. **Shard reassembly must be NUMERIC, not lexicographic.** The table ships as 128
   shards, `shard_0 .. shard_127`, each bf16 `[2500012, 160]`. Sorting the key strings
   gives `shard_0, shard_1, shard_10, shard_100, ...` and silently permutes a 95 GiB
   table. Verified against the checkpoint index: 128 keys, contiguous 0..127, and a
   lexicographic sort does produce that wrong order.

The forward itself is int64-exact and does NOT depend on Python bignum:
`layer_multipliers[i]` is a 0-dim int64 tensor, so the product is int64 arithmetic,
and it is bounded below 2^63 by construction because `multiplier_max * vocab_size <=
2^63 - 1`. **That bound holds only while every token id is below `vocab_size`.** An
out-of-range id overflows int64 and diverges silently, so the loader must not admit
one. Because `mixed` is therefore always non-negative, Python's `%` and C's truncating
`%` agree, and a port may use `int64_t %` without a sign correction.

Two further traps found in the cache path. The history of the previous
`ngram_size - 1` token ids is stored in the linear-attention cache as conv state 2 and
its dtype is **int64**, taken from the first tensor written; a port that stores it as
a float rounds token ids. And upstream's `update_conv_state` pads with **0**, which is
a valid token id, so the model works around it with an explicit EOS left-pad. Pad with
EOS, never with zero.

`split_ngram_parts` is **not used in the forward at all**. It is a checkpoint-layout
parameter consumed only by the weight conversion mapping, and saying so here stops the
next reader hunting for it in the model code.

**The PLE sits on decoder layer 1, not layer 2.** `ple_layer_ids` is 1-indexed and the
lookup is `config.ple_layer_ids.index(layer_idx + 1)`, so `[2]` selects 0-based layer
1. Confirmed from the checkpoint index: every PLE tensor is under
`model.language_model.layers.1.ple.`, and no other layer has one.

### PLE: a strided-history conv with no vLLM op, confirmed

**The dilated depthwise conv has no counterpart anywhere in vLLM, and the search is a
confirmed negative rather than an unfound one.** At `origin/main` = `6a5e8f5979`,
`git grep -in dilat` returns 17 lines tree-wide and **zero** in
`vllm/model_executor/layers/mamba/`, **zero** in `csrc/`, and **zero** in `tests/`.
`layers/conv.py` defines only `Conv2dLayer` and `Conv3dLayer`; there is no
`Conv1dLayer`, and the Transformers-backend auto-replacement maps only `nn.Conv2d` and
`nn.Conv3d`, leaving any `nn.Conv1d` as a bare PyTorch module. Upstream reached the
same conclusion from the other side and hand-rolled it, with the comment "We cannot use
the usual functions/kernels here for the short conv as the conv1d has dilation".

`causal_conv1d_fn` / `causal_conv1d_update` are disqualified on four independent
counts: they take no dilation argument; their Triton state loads unroll the taps at
unit stride in the kernel source; `state_len` is `width - 1` throughout the shape
plumbing where PLE needs `(width - 1) * dilation`; and vLLM has no `state_idx` concept,
so one layer cannot own three independently addressed conv states.

**The conv is strided history, not a local window.** `kernel_size=4`, `dilation=3`,
so output position `t` reads tokens at lags **{9, 6, 3, 0}** — a span of 10 tokens for
4 multiply-accumulates per channel, and the lag-0 tap makes it causal. The state is
therefore a genuine 9-deep ring buffer read at stride 3, and it cannot be compressed to
3 columns even though any single step touches only three of them.

Cost: 9 columns x 10240 channels = **~180 KiB per sequence at bf16 for this one
layer**. That is a real KV-budget line item, not a rounding error, and it belongs in
the `## Hardware` accounting once measured.

What the state holds is the **normed** conv input (`norm_conv`'s output), not the raw
hidden state and not the conv output. The layer forks: the skip term is the
**un-normed** `gated_value`, and only the normed copy enters the conv.

**The signed-sqrt gate has a trap in the clamp order.** It is
`gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`, so the clamp applies **before** the
square root and the floor on the output magnitude is `sqrt(1e-6) = 1e-3`, not `1e-6`.
Tiny scores are **amplified** to +/-1e-3 rather than squashed. Exactly zero maps to
zero, because `sign(0) = 0`, so the function is genuinely discontinuous at the origin
and that is reachable on a fully masked row. Mirror it; do not tidy it. A port that
clamps after the sqrt is wrong by three orders of magnitude in that band.

**A GDN state-length disagreement to reconcile at the seam.** Upstream sizes the GDN
conv state as `linear_conv_kernel_dim` = **4**, where vLLM's shape calculators use
`conv_kernel - 1`. The two conventions differ by one column, and nothing will announce
the mismatch.

**`ple_layer_ids` is one-indexed by design, not by accident.** The docstring says
"One-indexed", the config validator rejects ids outside `[1, num_hidden_layers]` and
resolves the layer type as `layer_types[layer_id - 1]`, and
`test_ple_layers_must_use_linear_attention` pins it. Do not "fix" it.

Padding is a paired obligation: the activations are masked **and** `ple_input_ids` has
its padded positions overwritten with EOS before the layer runs, because the n-gram
hash reads token ids rather than activations. Masking only the activations leaks
padding into the hash. `conv_mask` is `None` in steady-state decode, so the masking
lines are prefill-only.

One item is **AMBIGUOUS and must not be resolved from upstream**: the conv state
written during a chunked prefill whose first chunk is shorter than 9. Upstream
zero-pads on the left, which is arithmetically identical to what a single-shot prefill
would do, and its cache never reuses a prefix from another sequence. A prefix-caching
scheduler has to decide whether a cache hit restores the true 9-column state or re-pads
with zeros. That is our design question, not upstream's.

### Gated Residual: what our MHC actually gives us

The reuse verdict is sharper than "same shape, different math". Buffer plumbing is
largely reusable: the `[T, hc, H]` manifold, the layer-0 broadcast widen (upstream's
`hidden_states.repeat(1, 1, hc_count)` is exactly our broadcast), the per-token loop,
and the read/collapse/write-back cadence twice per layer. Three specifics:

- **`MhcPost` is bit-exact reusable for Qwen's write-back with the comb matrix set to
  identity.** The sum collapses to one non-zero term, so there is no reduction-order
  difference. It is 3x wasteful on the residual read and must not ship that way, but
  it gives a bring-up bridge from a kernel that is already gated, which is a free
  mutation target for the fresh reviewer.
- **`MhcSinkhorn` is dead here.** Qwen has no doubly-stochastic mixing and no comb
  matrix at all, so the carried `res_mix [T, hc, hc]` buffer goes with it. Qwen's
  streams couple only on the READ path, through the shared low-rank projection.
- `MhcPre` and `HcHeadCollapse` share a skeleton and no arithmetic: their norm is
  weight-free and global where Qwen's is grouped and weighted, their projection is one
  dense matrix where Qwen's is a two-stage low-rank `10240 -> 320 -> 10240`, their gate
  is per-stream scalar where Qwen's is per-element, and their reduce is a **sum** where
  Qwen's is a **mean**. Qwen also needs no separate head-collapse op: the final
  collapse is the same class with the injection branch switched off.

**`Qwen4ExpTextModel` has no final RMSNorm.** The mixer's own `hc_norm` is the last
normalization before `lm_head`. A port that copies our DeepSeek-V4 tail will insert one
that does not exist. Stated because that tail is the natural thing to copy.

**Weight parameterization differs from vLLM's op form.** Upstream applies
`output * (1.0 + weight)` with `weight` zero-initialized; vLLM's grouped norm applies
`out * weight` with `weight` ones-initialized. They coincide under a load-time
`w_vllm = 1.0 + w_hf`. Miss it and every `hc_norm` gets a near-zero scale, which reads
as a checkpoint bug rather than a port bug.

**The GGUF converter already folds it, and it folds far more than `hc_norm`.** Read at
source rather than relayed, because W5 writes the loader and the narrow version of this
sentence causes the defect it warns about. Every anchor below is read at our recorded
llama.cpp pin, stock upstream tag `b10451` (`10bf611e533d81f739128304991c5e133c6aebd8`,
[`../oracles/llama-cpp.md`](../oracles/llama-cpp.md)). Stock upstream has no `qwen4exp`
at all there (`git grep -il qwen4exp`: nothing tree-wide, so a released llama.cpp can
neither convert nor load this architecture). The converter is ggml-org/llama.cpp
[#27742](https://github.com/ggml-org/llama.cpp/pull/27742), head
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, **still OPEN**. Its `conversion/qwen4exp.py`
declares `class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase)`;
`_LinearAttentionVReorderBase` is `conversion/qwen.py:438`, a subclass of
`Qwen3NextModel` (`:365`, whose own signature is
`class Qwen3NextModel(_QwenMtpMixin, Qwen2MoeModel)`); and the PR's `modify_tensors` has
**no `hc_norm` branch**, so `hc_norm.weight` falls through to `super()`. The `+1` is the
inherited Qwen3-Next rule at `conversion/qwen.py:387-388`:

```python
elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
    data_torch = data_torch + 1
```

So the rule a loader implements is **every `*norm.weight` carries the fold, with
`linear_attn.norm.weight` (the GDN `ssm_norm`) the one exception** -- `hc_norm`,
`attn_q_norm` and `attn_k_norm` all match it, and the PLE and indexer gammas are folded
by the PR's own early-returning branch. A loader that skips the fold for `hc_norm` alone
double-folds everything else, which is the same silent ~2x defect one tensor to the left.
Two consequences for W5. The property belongs to one in-flight converter, not to "GGUF":
#27742 can change before it merges and another publisher's tool need not match it, so the
loader treats the fold as a provenance question and checks it -- cheaply, since an
unfolded `hc_norm` is a zero-init gamma and a folded one is centred on 1.0. And it was
corroborated on published artifacts during fresh review of #1988
(`unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ1_S` and `UD-Q4_K_XL`, `vumpt/...-Q4_K_M`, read
by HTTP range request against the bf16 HF tensors): every `*hc_norm.weight` is HF + 1.0
exactly, elementwise, while `ssm_norm` is unfolded and sits in [0.875, 1.023].

**Correction to the port map above.** vLLM's grouped RMSNorm is on **`RMSNormGated`**,
not the plain `RMSNorm`, whose only related knob is `var_hidden_size` -- a prefix
reduction that cannot express per-group norms. Verified directly: `RMSNorm` opens at
`layernorm.py:37` and `RMSNormGated` at `:172`, and the `group_size` parameter is at
`:187`. A porter reaching for `RMSNorm` finds nothing. Separately, `RMSNormGated`'s
`forward_cuda` dispatches to a flash-linear-attention Triton kernel rather than the
native reference, and that kernel's grouped numerics are **unverified**; whoever writes
the device arm owes that check.

The hyper-connection tower is **~640 M dense parameters** at this config (two modules
per layer x 48, plus the mixer), unquantized in the published scheme and read twice per
layer. That is a memory and bandwidth line item, not only a correctness one.

## Dependencies

Shared seams this row must route through rather than around, per AGENTS.md
"Shared seams". Each is named so a reviewer can check the routing instead of
inferring it.

- `ModelRegistry::Forward` and `dense_attn::AttnBlock` for decode.
- `vt::FusedChain` for model fusion; `layers::MlpGateUpMethodBase` and
  `vt::MergedGemmGroup` for the mergeable MLP projections.
- `include/vllm.h` for every shipped capability. Examples and servers stay thin
  ABI clients and never include an internal header.
- `vllm::HfConfigFromGguf` and the `qwen3_5` GGUF builder, which currently
  hard-asserts its own architecture and will refuse `qwen4_exp` by name until this
  row extends it.
- `src/vllm/v1/kv_cache_interface.*` for the third conv stream and the indexer side
  cache. This is the seam [#1963](https://github.com/mudler/vllm.cpp/issues/1963)
  and [#1966](https://github.com/mudler/vllm.cpp/issues/1966) are moving; coordinate
  rather than fork.

New files go beside their vLLM counterparts and mirror the upstream file structure,
per AGENTS.md. Where the upstream counterpart is MiniMax-M3 rather than a Qwen file,
mirror the op's home and say so in the file header.

## Work breakdown

Waves are separable and each is independently reviewable. Every wave lands reachable
from a production entry point, or names what is unreached with its owning row and
issue per AGENTS.md "Nothing lands dead".

- **W0, this pull request.** Spec, records, oracle exception proposal. No product
  code.
- **W1, config and registration.** `qwen4_exp` config resolution including the
  `full_attention` -> `qwen_sparse_attention` rewrite upstream performs in
  `__post_init__`, every `validate_architecture` rejection, and a refusal naming any
  unimplemented arm. Reachable through the loader.
- **W2, the two components with no vLLM op.** N-gram hashed embedding and the PLE
  layer with its dilated depthwise conv. Gated against transformers goldens on
  integer equality for the ID construction. First because they are the highest
  silent-wrongness risk and because they are independent of the attention work.
- **W3, gated residual.** The 10240-wide stream through the per-layer loop, both
  `use_combine` arms, and the final mixer.
- **W4, QSA.** Indexer side cache and KV spec, pooled-key build, block scoring and
  top-k, block-sparse consumer. Mirrors MiniMax-M3's op shape.
- **W5, assembly and the load plan.** Full model forward, vision path, MTP.
- **W6, the first runnable arm**, and the row's real unblock. Split by the blocker
  analysis above rather than by guesswork. **W6a** authors the `qwen4_exp` GGUF
  architecture -- one dispatch row plus its own config builder TU, never reusing
  `HfConfigFromGguf`, which asserts its own architecture by name -- and emits Q4_0 on
  every K=640 / K=320 reduction dim so the file can be opened at all. **W6b** is Route A:
  F16 table, mmap borrow, prefault off, CPU device, producing the token baseline.
  **W6c** is Route B: the dequantizing gather op plus the `kEmbeddingTable` keep-quant
  policy change, in that order, which is what makes the arm the developer actually chose
  reachable on CUDA.

Waves W2 through W4 have no ordering dependency on each other and can be dispatched
in parallel to separate worktrees. W5 is a barrier.

## Hardware

Usable budget on GB10 is about 119 GB. Read live from the HF API, 2026-08-26:

| Artifact | On disk | Verdict |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` BF16 | ~360 GB (`BF16 = 179,999,981,424` params) | no |
| `Qwen/Qwen3.8-Flash-Next-FP8` (official) | ~180 GB | no |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | ~128 GB; NVFP4 backbone with the n-gram table kept at **FP8, 51.2 GB** | no, over budget before KV |
| `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S | **67.56 GiB**, 3 shards | **YES, and it is the ONLY published artifact that does** |

**CORRECTED 2026-08-26.** This table previously read "README only, zero weight files
-- does not exist", and that was true when it was written and false a few hours later.
The repository was populated at 13:32Z with `UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-0000{1,2,3}-of-00003.gguf`,
72,546,461,344 bytes = **67.56 GiB**, read from the files' own headers:
`general.architecture = "qwen4exp"`, `split.tensors.count = 1224`,
`general.file_type = 24`. **It fits GB10 with roughly 52 GiB of headroom**, where every
safetensors artifact does not fit at all. The whole shape of this row's `## Work
breakdown` follows from that, which is why the correction is called out rather than
quietly applied.

Its metadata independently confirms this spec's own n-gram derivation to the digit:
`qwen4exp.ple.layer_multipliers = [23703573157769, 20109073645365, 8052911324071]`,
`qwen4exp.ple.head_vocab_sizes` starting `[20000003, 20000023, 20000033, ...]`, and
`qwen4exp.ple.layers = [1]` (0-based) corroborating the one-indexed conversion.

**Two things in our tree used to stop us loading it, and W6a
([#1989](https://github.com/mudler/vllm.cpp/issues/1989)) has since discharged both**
(`e228d6893`, #2019). When this row first read the file, our GGUF reader had no
`case 20`, so the IQ4_NL that file uses for `ffn_down_exps` and for the n-gram table
failed at header parse; and `KeepQuantKDim` returned `-1` for `kEmbeddingTable`, so a
quantized gather table expanded to bf16 and 51.2B params became 102.4 GB. W6a added the
IQ4_NL and Q5_0 reader arms and a dequantizing gather, and made `kEmbeddingTable`
keep-quant eligible. **The file opens today on the CPU arm.** What is still owed is the
CUDA half: `EmbeddingKernelCuda` decodes no blocks, so `DeviceQuantGatherSupported` is
true for `kCPU` alone and the table keeps its expand-bf16 residency on CUDA. `## Owed`
carries that, and it is where this model's high-concurrency advantage lives.

It carries **no MTP weights** — zero `nextn`/`mtp` tensors of 1224 — while the
safetensors repo has 31. That is [#1993](https://github.com/mudler/vllm.cpp/issues/1993)'s
problem and `docs/USAGE.md` must say so beside the arm.

**The revision is now PINNED, and only a local digest is still owed.** The repo's
`lastModified` moved again after this row first read it, which is exactly the
re-quantized-in-place case AGENTS.md "Say which weights, and from where" names; a repo
id alone is not a pin. `## Owed` now carries revision
`8bdc666649440e9bdc97e16f3f75782c98478ff5` and the three per-shard sizes and digests.
Those digests were the Hub API's `lfs.oid` values and were not locally computed. That
debt is now DISCHARGED: all three were recomputed with `sha256sum` on the staged copy on
29 August 2026 and agree three for three, recorded in
[the ladder-arm evidence file](../../docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md). The `split.tensors.count
= 1224` above is on the same footing and is recorded there as UNVERIFIED, because shard
1 is the metadata shard and reports `n_tensors = 0`.

llama.cpp still has no *merged* `qwen4_exp` architecture -- two competing PRs are open
or withdrawn -- so authoring our own converter remains owed for arms nobody publishes.

**The architecture hands us the lever.** Its card argues n-gram embedding is "more
amenable to offloading than MoE", and the arithmetic agrees: the per-token cost is
`(ngram_size - 1) * heads_per_ngram` = 16 lookups of `ple_embed_dim / ngram_heads` =
160 dims. **51.2B of the 180B parameters, 28% of the model, is a table touched 16 times per
token** (51.2 GB at FP8, 102.4 GB at bf16, ~31 GB at Q4_K_M). Making it non-resident is the intended design point. RadixArk reached the
same split independently.

| Arm | Backbone (125B) | N-gram (51B) | Resident | Fits |
|---|---|---|---|---|
| Q8_0 throughout | ~133 GB | ~54 GB | ~191 GB | no |
| Q4_K_M throughout | ~76 GB | ~31 GB | ~109 GB | yes, ~10 GB for KV and activations |
| **Q4_K_M backbone, n-gram table non-resident** | ~76 GB | 0 | **~76 GB** | yes, with room |

**Developer decision, 2026-08-26: the first runnable arm is the third row — a
Q4_K_M backbone with the n-gram table non-resident.** Q8_0 was raised and does not
fit: at ~191 GB it exceeds the budget by a wider margin than BF16 exceeds it on a
box half this size, and no partial-Q8 split reaches 119 GB while keeping the
backbone at 8 bits. Q4_K_M-throughout fits on paper at ~109 GB but leaves about
10 GB for KV and activations on a model whose native context is 262144, which is
not a margin. The chosen arm is also the only one that matches what the
architecture was built for, so the offload is a design point rather than a
concession.

This promotes the non-resident table from a note to a **first-class deliverable**
of W6. It is not free: GB10 is unified memory, so the existing host-pinned offload
seam (`ENG-WEIGHT-OFFLOAD`, mirroring vLLM's `cpu_offload_gb`) does not by itself
solve this there, and the mechanism has to be disk-backed or genuinely unloaded.
Establish that before designing around it.

These are sizing estimates from published parameter counts, not measurements. They
decide which arm to attempt first and nothing else. GB10 is **unified** memory, so
"offload to host" is not a move there; non-resident means disk-backed and page-cached,
and its cost is unmeasured. Establish it before it is designed around.

### The chosen arm has a hard blocker, and it is not the offload

Verified in this tree, 2026-08-26. The developer chose the Q4_K_M backbone with a
non-resident n-gram table, and that arm **does not load today**. The reason is not the
offload machinery and not the memory budget.

**This tree cannot keep a gather table quantized, by construction.** `KeepQuantKDim`
returns `-1` for `GgufTensorRole::kEmbeddingTable`, so the keep-quant branch is
unreachable for a gather table regardless of shape or encoding, and the qwen3_5 loader
asserts it by name: "the embedding table cannot keep quant blocks". A Q4_K or Q8_0
n-gram table therefore **expands to bf16 at load: 51.2B params become 102.4 GB of
anonymous memory** on a ~119 GiB box. The arm dies before the first forward. The reason
is already recorded in a header comment upstream of both -- "a gather, not a GEMM ... A
quantized-gather op is a follow-up row" -- and **no such row exists**. That sentence is
the whole blocker and it has been sitting in a comment.

The only non-expanding residency for a gather table is `kKeepF16`, which requires the
file to store ggml type **1 (F16) exactly**. That makes the table 102.4 GB on disk, and
it is **CPU-only**, because `EmbeddingKernelCuda` refuses anything but f32/bf16.

**Second blocker, cheap to avoid because we author the converter.**
`moe_intermediate_size = 640` makes `ffn_down_exps` Q4_K-illegal on its reduction dim
(`640 % 256 = 128`), and `hc_lowrank = 320` is the same class. llama.cpp's substitution
for a ragged-K Q4_K tensor is **Q5_0, now VERIFIED** and no longer owed: the
`tensor_type_fallback` table in `src/llama-quant.cpp` maps `Q4_K -> Q5_0`,
`Q5_K -> Q5_1`, `Q6_K -> Q8_0`, `Q2_K/Q3_K/TQ* -> Q4_0` and every `IQ*` including
`IQ4_XS -> IQ4_NL`, then falls to `F16` if the result still does not divide. So the
answer depends on the RECIPE, which is why the shipped `unsloth` UD-IQ1_S file shows
`IQ4_NL` on `ffn_down_exps` rather than Q5_0 -- it asked for an IQ type, not Q4_K. A
`-Q4_K_M` build of this model would land on Q5_0, and on Q8_0 wherever `use_more_bits`
promotes `ffn_down` to Q6_K. The
dependent fact IS verified in-tree and is the one that bites: this repository's GGUF
reader knows ggml type ids `0,1,2,8,10..14,16,18,19,22..28,30,39,40,41,66` and **has no
entry for 3 (Q4_1), 6 (Q5_0), 7 (Q5_1) or 20 (IQ4_NL)**, so such a file fails at header
parse with "unknown ggml type id". A stock `llama-quantize -Q4_K_M` output for this
model would not open at all. The fix is ours: emit **Q4_0** on every K=640 and K=320
reduction dim -- block 32, the same 4.5 bpw as Q4_K, and keep-quant capable.

**`ENG-WEIGHT-OFFLOAD` will not help, now or later.** It moves zero bytes today
(`ConsiderWeight` has no production callers, `supports_weight_offload` is false
everywhere and a test pins that), and it is separately documented inert on GB10 because
it moves bytes inside one physical pool. **Do not budget for it.**

**The tier that does work already ships**, and the 2.4T model is the proof: mmap the
GGUF `MAP_PRIVATE`, borrow tensors in place, and alias the host pointer into the kernel
on a host-addressable device. That serves 369.97 GiB from a 119.631 GiB box at ~62 GiB
resident. Set `vllm_cpp.mmap.prefault: false`, or `PrefaultBorrowedSpan` touches every
page of the table at load and OOM-reboots the box.

**Two routes, and the recommendation is to do both in order.**

- **Route A, runs on today's code, CPU only.** F16 n-gram table, Q4_0 on the ragged
  reduction dims, Q4_K elsewhere, mmap borrow with prefault off. Delivers a correct
  first run and the token baseline Route B needs. No shared-kernel changes.
- **Route B, the arm actually chosen.** Add a dequantizing gather to `vt::Embedding`
  across CPU and CUDA, then make `kEmbeddingTable` keep-quant eligible gated on that
  op's availability. Order matters: the assertion above is CORRECT today and only
  becomes wrong once the op lands. Then the table is Q4_K at **28.8 GB** on disk,
  borrowed, device-aliased, gathered on device -- smaller than the 51 GB the arm was
  scoped at. Roughly 400 lines.

**Corrected sizing.** Backbone ~67.7 GiB resident in the expected arm; whole process
~73.5 GiB of 119.631 at 32K context single stream, leaving ~46 GiB of headroom that is
exactly what pays for the table's page cache. The original ~76 GB estimate was right to
within 10%. The design works because the per-token demand is tiny: 16 lookups x 160
dims x 2 B = 5120 B/token over at most 16 distinct pages, so **<= 64 KiB of reads per
token**, against the 2.4T expert lane's 6.95 GB/token. That contrast is the whole
argument for this arm, and it is why the table is offloadable where MoE experts are not.

Two further hazards, both with escapes: the `--device cuda` load-time device-fit
refusal counts every tensor in the file including the table, and a misaligned mmap
borrow is silently STAGED into device memory with a full `Alloc` -- pad the n-gram
tensor's data offset to 256 in our writer, since `kDeviceAliasAlignment` is 256 while
GGUF guarantees only 32.

Finally, **there is no GGUF writer in this repository.** Authoring the conversion means
authoring it outside this tree; what this repo controls is only what it will accept.
And a latent trap for exactly that writer: the parse-time and dequant-time divisibility
checks test `numel % block_elems`, not `K % block_elems`, so a hand-rolled ragged-K
K-quant tensor decodes across row boundaries into structurally wrong values with **no
error**. Assert K-divisibility in the converter.

## Risks

- **Porting the eager reference as written.** The stated risk of the oracle split.
  The QSA indexer and the n-gram ID construction are both written as scalar Python
  in transformers. A reviewer should mutate for this: an implementation whose QSA
  path has no block-level kernel is a correctness result, not a port.
- **Reaching for DSA.** This tree has a working DSA indexer, and it is the wrong
  base. See above.
- **Silent n-gram mis-indexing.** See above.
- **Sizing estimates hardening into measurements.** The `## Hardware` table is
  arithmetic on published counts. `.agents/` already records this failure mode
  (a quoted number becoming a measured one); do not let the 76 GB row be cited as
  an observation.
- **A GGUF arm with no oracle.** llama.cpp does not implement `qwen4_exp`, so the
  usual quant-arm cross-check against a quant-matched llama.cpp does not exist. A
  k-quant arm here can only be gated against our own higher-precision path, which
  is a weaker gate, and the spec must say so rather than imply parity.
- **`transformers_version: 5.8.0.dev0`** in the published config is older than both
  our pin and the branch that merged `Qwen4Exp`. It records the branch the config
  was authored on and is not a usable pin. Do not resolve the oracle from it.

## Tests to port

AGENTS.md requires the upstream tests in the same change, preserving parameters,
modes, fixtures, tolerances, failure cases and the revision anchor. The upstream
suite is `tests/models/qwen4_exp/test_modeling_qwen4_exp.py` at transformers #48337,
707 lines, two classes. Inventory, read live 2026-08-26:

**`Qwen4ExpTextModelTest`** (`Qwen4ExpTextModelTester`, a `CausalLMModelTester`):

| Upstream case | Ports to | Note |
|---|---|---|
| `test_ple_layers_must_use_linear_attention` | W1 | a config invariant; cheap and load-bearing |
| `test_ple_padding_and_static_cache_match_unpadded_sequence` | W2 | the padding/EOS-segment semantics of the n-gram history |
| `test_all_layer_types_cached_forward_match_full_forward` | W4/W5 | cached vs full forward across BOTH layer types; this is the incremental-decode gate |
| `test_ple_beam_generation` | W5 | PLE under beam search, where the conv and n-gram states must follow the beam |
| `test_ple_sharded_checkpoint_loads_and_forwards` | W5 | 131 shards here, so sharded load is not optional |
| `test_generate_with_ple_and_inputs_embeds` | W5 | drives `reverse_embedding`, the inputs-embeds path |
| `test_reverse_loading_mapping` | W1 | weight-name mapping both directions |
| `test_attention_outputs`, `test_hidden_states_output` | W3/W4 | both are OVERRIDDEN upstream because the hyper-connection stream changes the shapes; port the override, not the base |
| `test_tp_plan_matches_params` | not ported | tensor-parallel plan; no TP surface in this row |
| `test_generate_compile_model_forward_fullgraph`, `test_generate_compilation_all_outputs`, `test_multi_gpu_data_parallel_forward`, `test_generate_with_quant_cache` | not ported | torch.compile / multi-GPU / torch quant-cache harness, no counterpart here |

**`Qwen4ExpCompositeModelTest`** (`Qwen4ExpVisionText2TextModelTester`, a
`VLMModelTester`): `test_mismatching_num_image_tokens`, `test_video_forward`,
`test_composite_checkpoint_loads_as_causal_lm`,
`test_base_model_checkpoint_loads_as_conditional_generation`,
`test_generate_with_ple_and_inputs_embeds`, plus its own `test_attention_outputs` /
`test_hidden_states_output` overrides. All port to W5. The remaining cases in that
class are the same harness-only skips as above.

Adaptations must be documented per AGENTS.md, and only where genuinely unavoidable.
"Our harness differs" is not one; "upstream asserts against a `torch.compile`
fullgraph we do not have" is.

### Local red-first tests

Red-first, smallest failing test per slice, each entering through a production entry
point per AGENTS.md "Nothing lands dead". A unit test that constructs the type by
hand does not discharge this.

1. N-gram ID construction against transformers goldens: prime head vocab sizes, the
   splitmix64 multipliers, the shift-and-XOR mix, EOS segment handling. Integer
   equality, no tolerance.
2. Grouped RMSNorm against vLLM's `group_size` form.
3. Gated Residual forward against transformers, both `use_combine` arms.
4. QSA block selection: selected token index sets equal to transformers on the same
   inputs, including the ragged tail beyond the last complete block.
5. PLE layer end to end, including the dilated depthwise conv and its state.
6. Config resolution: the `full_attention` -> `qwen_sparse_attention` rewrite that
   upstream `__post_init__` performs, and every rejection in `validate_architecture`.
7. Loader coverage against the published index, with the refusal path naming any
   unimplemented arm.
8. Inertness: existing Qwen3.5/3.6/3.8 goldens byte-identical.

## Gates

No token gate is claimable until an arm runs. In order:

1. **G0, component goldens.** Tests 1-6 above against transformers at the lane pin.
   This is the only gate reachable today, and it is reachable without the weights.
2. **G1, load plan.** Every published tensor accounted against a committed manifest,
   per arm, with refusals naming what is missing.
3. **G2, token-exact greedy** with **at least one prompt past `indexer_budget` = 2048
   tokens of context**, because below that QSA selects every candidate and the gate
   cannot distinguish a correct implementation from one attending pooled keys. Vs
   transformers at the lane pin, on whichever arm
   `## Hardware` makes runnable first. Strict token equality; the near-tie
   distributional doctrine applies only if the oracle's greedy decode is shown
   non-deterministic, which is not assumed here.
4. **G3, quantized arms.** Per arm, with the lower-bound requirement this repository
   places on quantized gates, and with the missing-llama.cpp-oracle limitation stated
   in the result rather than omitted.
5. **G4, speed against llama.cpp at its pin.** A denominator now exists and the
   earlier "there is no denominator" clause is superseded. **The oracle is
   [`llama-cpp-qwen4exp`](../oracles/llama-cpp-qwen4exp.md), pinned at
   `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3` (ggml-org/llama.cpp PR #27742), and NOT
   the stock [`llama-cpp`](../oracles/llama-cpp.md) pin this clause used to name.** That
   correction is measured, not stylistic: `llama-cpp` is pinned at released `b10451`,
   and a tree-wide grep for the string qwen4exp at that revision matches nothing (rc=1)
   while the same grep for qwen3vl matches three or more files (rc=0) as the control.
   Both were run against a fresh bare clone and are recorded in
   [the oracle file](../oracles/llama-cpp-qwen4exp.md) and in
   [#2060](https://github.com/mudler/vllm.cpp/issues/2060); they are not repeated as a
   command here, because a grep against a llama.cpp revision is not something this
   tree's gate runner can execute and a gate item that cannot fail gates nothing. The
   released oracle cannot name this architecture, so it cannot supply this denominator,
   and a gate that named it was naming a tool that refuses the model.

   `llama-cpp-qwen4exp` reads `gateable = yes` as of 29 August 2026, and both halves are
   recorded: it builds on CUDA for GB10 and `llama-server` at the pin loaded
   `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S and returned 64 coherent greedy tokens.
   [#2060](https://github.com/mudler/vllm.cpp/issues/2060) owed that run half and is
   discharged; the evidence is
   [`docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md`](../../docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md).
   **W6a has since made that file loadable on our side** (#2019), so the encoding
   precondition is met on the CPU arm; the gate itself still waits on G2, and no
   throughput, latency or memory number is admissible from this row until G2 passes.

   **`gateable = yes` is not a measurement.** That oracle's own record says so: one
   prompt, one repetition, five prompt tokens, one slot, no clock window and no
   contention control. Nothing in that evidence file may be quoted as a number.

   **The target is binding.** The developer's words, 2026-08-26, quoted rather than
   paraphrased because the wording is the requirement:

   > we should be faster than llama.cpp

   > especially at high concurrency

   Therefore:

   - **A concurrency LADDER is the headline, not a point.** c = 1, 4, 8, 16, 32 at
     minimum. A c=1 result neither confirms nor refutes this target. The harness walks
     the published online-serving grid, which is that set plus the c=2 the grid carries:
     `(1, 6) (2, 6) (4, 12) (8, 24) (16, 96) (32, 192)` as (concurrency, prompts).
   - **1,024 input tokens and 128 output tokens, three repetitions.** Stated here
     because this section used to be silent on the lengths while the runs used the
     published grid's values anyway, and an unstated input length is the axis that
     splits prefill from decode. These are `online_gate.INPUT_LEN`, `OUTPUT_LEN` and
     `REPETITIONS`, copied rather than chosen so a cell of the llama.cpp table can be
     read beside a cell of `docs/benchmarks/vllm-online-serving.md`.
   - **Prefill and decode reported separately**, because input length splits them and an
     aggregate hides which lever moved.
   - Memory is an axis: peak RSS and peak device bytes at each concurrency.
   - llama.cpp runs in its production configuration. A handicapped denominator is not a
     result, and this repository already has the `--enforce-eager` precedent for how that
     goes wrong.
   - Identical artifact, prompts, token counts, sampling and concurrency; idle host;
     reproduced with a same-binary A/B. "Identical prompts" means the same
     per-(concurrency, repetition) corpus partitions
     `online_gate.prepare_corpus_views` writes and refuses to let overlap — not one
     shared file replayed three times on one side against three disjoint thirds on the
     other — and the same `--num-warmups`, which `OnlineRun.num_warmups` sets to the
     concurrency.
   - **The denominator is TEXT-ONLY and this row is a multimodal port.** `/props` on the
     llama.cpp arm reports `"modalities":{"vision":false,"video":false,"audio":false}`.
     An arm that loads a vision tower does strictly more work per request, so the
     vllm.cpp cell paired against it must be a text-only configuration on the same
     UD-IQ1_S artifact. State which side ran what, or the ratio measures a configuration
     difference and reads as a performance one.

### Where the speed is expected to come from, and what would forfeit it

Four levers, from a source study of the two llama.cpp implementations (#27742 open,
#27739 closed by courtesy). **Both are UNMERGED**; each item is a reading of a pinned SHA
and not a measurement. Three of them constrained waves that had not started when this was
written; W3, W4 and W6a have since landed, and each lever below now records what its wave
actually did rather than what it was asked to do.

1. **Continuous batching and paged KV — the concurrency lever.** This engine mirrors
   vLLM's scheduler and block manager; llama.cpp's server allocates fixed parallel slots.
   That gap grows with concurrency rather than shrinking, which is where the target aims.
2. **The QSA consumer — the long-context lever, and the one this row could have
   forfeited by accident.** #27739 records that a sparse **mask** over a dense cache costs the same as
   dense attention under CUDA flash attention, because `flash_attn_mask_to_KV_max` only
   scans back to the first tile that is not all `-inf`. #27742 is mask-only and so buys
   correctness without decode speed. **W4 built the gather, not the mask** (#2030): the
   consumer counts at the key-row read, so the lever is preserved rather than forfeited.
3. **N-gram table residency.** #27742 makes the table CPU-resident by tensor class
   regardless of `-ngl`, so every token's 16 gathers are host work. At IQ4_NL the table is
   ~28.8 GB inside a 67.56 GiB file against ~119.6 GiB usable, so it can be
   device-resident and quantized. At batch B that is 16xB uncoalesced random gathers, so
   host-versus-device here is a scaling difference, not a constant. **W6a asserted that
   decision rather than defaulting it** (#2019): the table stays quantized and is gathered
   in place on CPU. The CUDA arm is unbuilt, so on CUDA the table still expands, and this
   lever is only half collected.
4. **The hyper-connection write-back.** Both PRs materialise the rank-1 update as a
   `repeat_4d` + `mul`: 96 materialised `[2560, 4, T]` broadcasts per forward at 48
   layers x 2 sites. **W3 landed leaving the fused seam reachable rather than built**
   (#2045): `GatedResidualWriteBackInPlace` is the primitive, and no device kernel
   replaces it yet, which `## Owed` carries.

**No ceiling may be declared** if a first measurement disappoints. An apparent
same-artifact limit is an unresolved implementation difference with a next traceable
hypothesis, every time.

## Evidence required

Per gate: the exact build and run recipe, the lane transformers revision, the
checkpoint repo **and revision** plus sha256 for any quantized artifact, the device,
and the contention state. `docs/USAGE.md` gains the checkpoint pins in the same
change that makes any arm reachable, not later.

## Mutation record — W6a (#1989)

Committed because the first fresh review could not re-run W6a's claimed
mutations: no table for the wave existed anywhere in the tree, so the reviewer
designed and ran fourteen of their own. This section is the reproducible list.
Every row is one textual change applied to a pristine tree, rebuilt, run,
restored, and rebuilt again with the source `touch`ed after restore — without
that touch ninja skips the rebuild and the mutations ACCUMULATE, which fails
toward RED and makes a weak gate read strong.

Reviewer battery (14, at `beedfdf31`; R8b and R11-R13 are what the review's
findings F2 and F7 are made of):

| # | mutation | target(s) | result |
|---|---|---|---|
| R1 | `kValuesIq4nl[8]` `1` -> `0` | dequant, embedding | RED, RED |
| R2 | `DequantQ5_0` upper-half `qh` shift `j+12` -> `j+16` | dequant, embedding | RED, RED |
| R3 | `DequantIQ4_NL` swap the two nibble halves | dequant, embedding | RED, RED |
| R4 | reader `GgmlTypeTraits` IQ4_NL `block_bytes` 18 -> 17 | load_plan, traits | RED, RED |
| R5 | `vt` `BlockGeometry` Q5_0 `block_bytes` 22 -> 21 | traits | RED |
| R6 | delete the block arm of `EmbeddingKernel` | embedding, qwen36_loader | RED, RED |
| R7 | `KeepQuantKDim(kEmbeddingTable)` back to `-1` | keep_quant, qwen36_loader, load_plan | RED x3 |
| R8a | `DeviceQuantGatherSupported` INVERTED | keep_quant | RED |
| R8b | `DeviceQuantGatherSupported` widened to every device but ROCm | keep_quant | SURVIVED — only the CPU branch is reachable on a CPU host, the same limitation `DeviceKeepQuantSupported` already has |
| R9 | remove the NVFP4 `role != kEmbeddingTable` exclusion | keep_quant | RED |
| R10 | delete the `kGgufArchArms` `qwen4exp` row | model_loader_gguf | RED |
| R11 | neuter `vt::Embedding`'s whole-block precondition | embedding | SURVIVED at `beedfdf31` -> **RED after the F7 repair** |
| R12 | `VecDotIQ4_NLQ8_0`: swap the two nibble halves | all 8 suites | SURVIVED x8 at `beedfdf31` -> **RED after the F2 repair** |
| R13 | `VecDotQ5_0Q8_0`: upper-half `qh` shift `j+12` -> `j+16` | all 8 suites | SURVIVED x8 at `beedfdf31` -> **RED after the F2 repair** |

Repair battery (this change; each restored byte-identically and re-verified
green afterwards):

| # | mutation | target(s) | result |
|---|---|---|---|
| R11 | neuter `vt::Embedding`'s whole-block precondition (`% BlockElems` -> `% 1`) | `test_ops_embedding_quant` | RED |
| R12 | `VecDotIQ4_NLQ8_0`: swap the two nibble halves | `test_ops_quant_dot` | RED |
| R13 | `VecDotQ5_0Q8_0`: `>> (j + 12)` -> `>> (j + 16)` | `test_ops_quant_dot` | RED |
| R14 | `NoKeepQuant` made a no-op (the F1 defect, restored) | `test_deepseek_v4_gguf_load`, `test_laguna_gguf_load` | RED, RED |
| R15 | delete the block arm's per-id bounds check (`id % v`) | `test_ops_embedding_quant` | RED |
| R16 | `ResidentWeight`'s CPU alias offset by one byte | `test_gguf_qwen36_loader` | RED |

Anchor repairs in W6a: **three**, not nine. Measured with the repository's own
checker on both trees — parent `ok=876, stale=31, broken=6 -> rot 37`; head
`ok=879, stale=28, broken=6 -> rot 34`. The three are
`KERNEL-ATTN-DFLASH-BLOCK -> cpu_ops.cpp`, `SPEC-DFLASH-GGUF -> :773 -> :1015`
and `SPEC-MTP-GGUF -> :971 -> :1425`. All three were stale BEFORE W6a. The
DFlash one landed with a label that disagreed with its own href and is corrected
here.

## Mutation record — W5a (#2031)

The W5a pull request claimed a "14/14 red" battery. **That claim was
inaccurate and is not repeated here.** The fresh review of `a68312c79` re-ran
it, found two survivors, and this section is the honest list — every survivor
kept in the table rather than dropped, because a mutation table whose only rows
are reds is a table nobody re-ran.

Method, unchanged from the W6a section above: one textual change applied to a
pristine tree, `touch`ed, rebuilt, run, restored, `sha256sum`-verified
byte-identical against the pre-mutation copy, `touch`ed again and rebuilt. Every
row below was measured on this branch by the W5a REPAIR, not relayed, and every
row of the repair battery was re-measured on the FINAL head rather than at the
point in the repair where its fix landed — the case COUNT moves as cases are
added, and a count carried forward from an earlier build is a number nobody
measured.

**Reviewer battery at `a68312c79` (the immutable head), reproduced by the
repair before changing anything:**

| # | mutation | target | result at `a68312c79` |
|---|---|---|---|
| M1 | delete the `LoadQwen4ExpFromGguf` call site in `qwen4_exp_registry.cpp` (return `Qwen4ExpWeights{}`) | `test_qwen4_exp_gguf_weights` | **PARTIAL** — 1 of 10 cases red (6 assertions). Only "a malformed file refuses BY NAME" reddened; the headline case "the production load_weights hook LOADS the file" stayed GREEN, because `REQUIRE_NOTHROW` + `model != nullptr` is satisfied by a stub |
| M5 | swap `g` and `t` in `qwen4_exp_weights.cpp`'s `ReorderVRows`/`ReorderVCols` | `test_qwen4_exp_gguf_weights` | RED, 2 of 10 cases, 41 assertions |
| M6 | the SAME swap in `qwen3_5_gguf_weights.cpp`'s copy | `test_gguf_qwen36_loader` 7/7 555, `test_model_loader_gguf` 7/7 23, `test_gguf_nvfp4` 14/14 2352, `test_gguf_keep_quant` 42/42 6340 | **SURVIVED x4** — the cause is the FIXTURES, not the loader, and the cause first recorded here was itself wrong. Measured in `tests/vllm/test_gguf_qwen36_loader.cpp`: the fixture DEFAULT is `num_k = 2, num_v = 2` (`:120`), written out as `ssm.group_count = 2, ssm.time_step_rank = 2` (`:149-150`), so `K = 2` and `R = 1` and the permutation is the IDENTITY — the reorder is inactive. The ONE case that reaches it at all, `TEST_CASE("LoadQwen3_5MoeFromGguf: V-head reorder when num_v != num_k")` (`:361`), sets `num_v = 4`, giving `K == R == 2`, where the permutation is its own INVERSE and the mutated loader emits byte-identical weights. So exactly one case exercises the reorder, at the one shape where forwards and backwards are indistinguishable. The other three suites declare no `ssm.group_count` at all, so no buffer passes through the reorder there. Owned by #2081 |
| MUT-C | delete `text["ple_embed_dim"] = ple_row * ngram_heads_gguf;` from `Qwen4ExpHfConfigFromGguf` | `test_qwen4_exp_gguf_weights` | **SURVIVED** — 10/10, 2938/2938. The fixture defined `kPleRow` as `kH / kNgramHeads`, so `ple_row * ngram_heads == hidden_size` BY CONSTRUCTION and the fallback was indistinguishable from the value |

**Repair battery (this change). Each restored byte-identically, sha256-verified,
and re-run green afterwards:**

| # | mutation | target | result AFTER the repair |
|---|---|---|---|
| M1 | delete the `LoadQwen4ExpFromGguf` call site | `test_qwen4_exp_gguf_weights` | **RED, 2 of 11 cases, 8 assertions** — the headline case now opens the handle with `ModelAs<Qwen4ExpLoadedModel>` and reads loaded tensor bytes, which a stub cannot produce |
| MUT-C | delete the `ple_embed_dim` line | `test_qwen4_exp_gguf_weights` | **RED, 9 of 11 cases, 7 assertions** — the fixture's `kPleRow` is independent of `kH`, so the total is not the `hidden_size` fallback and the PLE projections refuse by shape. Re-measured at `kPleRow = 96` by the W5a-3 repair: still RED, 9 of 11, 7 assertions of 2553 run |
| MUT-G1 | disable the new `DeviceQuantGatherSupported` guard (`if (false && ...)`) | `test_qwen4_exp_gguf_weights` | **RED, 1 of 11 cases, 8 assertions** — "a device with no block gather refuses BEFORE the load" |
| MUT-G2 | pin the production device ARGUMENT to `vt::DeviceType::kCPU` in `qwen4_exp_registry.cpp` | `test_qwen4_exp_gguf_weights` | **SURVIVED, and it cannot do otherwise on this host.** A CPU-only build registers no other platform, so `CurrentPlatform().device_type()` and the literal `kCPU` are the same value and no test can tell them apart. The GUARD is gated (MUT-G1); the ARGUMENT is not, and this row exists so nobody records that as coverage. Closing it needs a CUDA host |
| M5 | swap `g`/`t` in our reorder copy | `test_qwen4_exp_gguf_weights` | RED, 2 of 11 cases, 41 assertions |
| M6 | the same swap in the `qwen3_5` copy | the same four suites, re-run at THIS head | **SURVIVED x4, deliberately unfixed.** Filed as [#2081](https://github.com/mudler/vllm.cpp/issues/2081): re-shaping a shipped model's fixtures changes `qwen35`, `qwen35moe` and `qwen3next` coverage and is not this row's scope. The source comment and the `## Owed` entry that both claimed "gated on both sides" are corrected to say so |

**Two survivors stand at the end of this repair, and both are named rather than
closed:** M6 (owned by #2081) and MUT-G2 (owned by the CUDA gather arm under
[#2083](https://github.com/mudler/vllm.cpp/issues/2083), which needs a device
this repair did not have).

**W5a-3 battery (this change).** Three repairs: the missing `issue-index.md` row
for #2081, the residual `kPleRow`/`kH` coincidence the re-review found, and the
`Closes #2064` note below. Each mutation was applied to a pristine tree,
`touch`ed, rebuilt, run, restored from a byte-identical copy,
`sha256sum -c`-verified, rebuilt and re-run green.

| # | mutation | target | result |
|---|---|---|---|
| M6 | swap `g` and `t` in `qwen3_5_gguf_weights.cpp`'s `ReorderVRows`/`ReorderVCols` | the four suites, re-run at THIS head | **SURVIVED x4, re-measured not relayed** — `test_gguf_qwen36_loader` 7/7 555, `test_model_loader_gguf` 7/7 23, `test_gguf_nvfp4` 14/14 2352, `test_gguf_keep_quant` 42/42 6340, every count identical to the un-mutated baseline. This is the measurement the appended #2081 index row states |
| M5 | swap `g` and `t` in OUR `qwen4_exp_weights.cpp` copy | `test_qwen4_exp_gguf_weights` | **RED, 2 of 11 cases, 41 of 2970 assertions**, re-measured at this head. The pair M5/M6 is what makes "only one copy is gated" a measurement rather than a reading |
| MUT-D | `text["ple_embed_dim"] = ReqInt(gguf, p + "embedding_length") * ngram_heads_gguf` in `Qwen4ExpHfConfigFromGguf` — `hidden_size` where the per-head row width belongs | `test_qwen4_exp_gguf_weights` | **SURVIVED at `kPleRow = 64`** — 11/11, 2969/2969. `kH` is also 64, so `hidden_size * ngram_heads` and `ple_row * ngram_heads` were the same 128 and the wrong source was unobservable. This is the residual coincidence the re-review named, and the reason MUT-C alone did not close #2064 |
| MUT-D | the same mutation AFTER `kPleRow` moved to 96 | `test_qwen4_exp_gguf_weights` | **RED, 9 of 11 cases, 7 of 2553 assertions.** At 96 the correct total is 192, the `hidden_size` product is 128 and the bare `hidden_size` fallback is 64: all three distinct, so each wrong source refuses the file by shape |

96 is the smallest legal replacement. `head_dim_per_ngram() == kPleEmbedDim /
kNgramHeads == kPleRow` must stay a whole number of Q8_0 blocks, because the
n-gram table is the one gather this model keeps quantized, so `kPleRow` is a
multiple of 32; 32 is `kH / kNgramHeads` and 64 is `kH`, and 96 is the next one.
A third `static_assert` now pins `kPleEmbedDim != kH * kNgramHeads` beside the
two that were already there, and the suite carries a third `CHECK` on
`ple.embed_dim` so both wrong answers are visible to a reader, not only to a
mutation.

**#2064 closes when W5a lands, and the closing keyword lives in the pull request
body.** The wave fixed it in flow — MUT-C and both MUT-D legs are its
instruments — but the issue is still open, and this branch has no pull request
yet. Whoever opens it puts `Closes #2064` in the BODY, which is the landed commit
message here (`squash_merge_commit_message = PR_BODY`), so the merge closes the
issue. Do not close it by hand: a hand-closed issue leaves the commit that fixed
it unlinked, and that link is the only thing tying the fix to its record.

## Mutation record — W5b-1 (#2110)

The cross-TU GDN seam (`RunGdnBlockPaged` / `BuildGdnStepInputs`). Every row
below was measured by the W5b-1 REPAIR on the merged head, not relayed from the
implementer or from the fresh review, because the review found that the case's
own comment named an outcome no mutation of that function can produce.

Method, unchanged from the W5a section above: one textual change applied to a
pristine `src/vllm/model_executor/models/qwen3_5.cpp`, rebuilt with the BUILD RC
read before any test result, run, restored from a byte-identical copy,
`sha256sum -c`-verified at
`d0db911160f326ab83e3fdc13ee8dd4df0f25a9f292790eb1ccf3a08a087cdfc`, rebuilt and
re-run green. The un-mutated baseline on this head is
`test_qwen3_5_gdn_spec_routing` 7 cases / 82 assertions,
`test_qwen27_paged_forward` 31 / 770, `test_qwen35_moe_gdn_ba_owner` 1 / 23,
`test_qwen3_5_decode_graph_seam` 10 / 156.

| # | mutation | target | result |
|---|---|---|---|
| A | perturb the gated-RMSNorm epsilon inside `GdnBlockPaged` (`+ 1e-3F` at its one definition) | `test_qwen3_5_gdn_spec_routing`, `test_qwen27_paged_forward` | **RED on the EXISTING cases: 1 of 7 (the MIXED spec+prefill case, 2 assertions) and 5 of 31 (6 assertions).** The Qwen3.5/3.6 forward still runs this block, so the block the wrapper exposes is the live one. **The NEW seam case stays GREEN, and it must:** both arms of its comparison enter `GdnBlockPaged`, so no uniform mutation of that function can separate them. The implementer's comment claimed this mutation reds the new case together with the spec-routing cases; that claim is false and is corrected in the test |
| B | make the wrapper stop delegating — allocate a `[T,H]` output at `GdnOutDType()` and `Zero()` it instead of calling `GdnBlockPaged` | `test_qwen3_5_gdn_spec_routing` | **RED on exactly the new case, 10 assertions**, the 6 pre-existing cases green. Six are the output/SSM/conv bit-for-bit comparisons at both gate dims; four are the `dh_fp8` sub-case, which now sees the wrapper RETURN instead of refusing. This is the mutation that reds the new case |
| C | make the wrapper stop FORWARDING `dh_fp8` (`GdnBlockPaged(..., T, nullptr)`) | `test_qwen3_5_gdn_spec_routing` | **RED on exactly the new case, 2 of 82 assertions.** Before the repair this mutation SURVIVED at 7 / 74: the comparison case runs a BF16 weight set where `ProjectGdnQkvz` never reads `h_fp8`, so the only production argument the wrapper forwards that nothing gated was the fp8 one. The sub-case added by this repair closes it |

**Why `dh_fp8` is gated by a refusal and not by a number.** `dh_fp8` selects
between two mutually exclusive production leaves inside `ProjectGdnQkvz`:
non-null takes `MatmulFp8CutlassPreQuantD`, null takes `MatmulFp8CutlassD`. Both
open with a `VT_CHECK` on `vt::OpRegistered(kMatmulFp8CublasLt, device)`, and
that op is registered for `kCUDA` alone
(`include/vllm/model_executor/models/dense_fp8_gemm.h`), so on a host queue the
fp8 GDN tower computes no value the case could compare. It does produce a
refusal that names the leaf that refused, and the two leaves name themselves
differently. BOTH directions are asserted, so the observable discriminates
rather than merely observes. A CUDA host can strengthen this to a numeric
comparison; that is owed, not done here.

**Mutation B's build is the trap this record exists to name.** Removing the
delegation leaves `gdn_meta` and `state` unused, and this tree builds with
`-Werror=unused-parameter`, so the first attempt does not compile — and a
mutation that never built leaves the STALE binary printing green. Read the build
RC before any test result. The fresh review hit exactly this and read a false
7 / 74 pass.

## Mutation record — W5b-3 (#2156)

The PLE dilated depthwise conv as `vt::Qwen4ExpPleConv`. Sixteen mutations, one
at a time, each proved APPLIED by a sha256 that moved, each build's exit status
read BEFORE any test result, and the tree restored byte-for-byte and re-verified
by sha256 after every one. Re-measured on the final head. Suites:
`test_qwen4_exp_ple_device` (the new device gate, 10 cases / 538 assertions
green) and `test_qwen4_exp_ple` (the W2 host suite, 9 / 395 green), the second
present as a control that no mutation of the device arm can move.

**Five of the sixteen failed to BUILD on the first pass, and that is a result
about the harness rather than about the code.** `-Werror` turns "the mutation
made a variable unused" into a link that never happens, the runner then executes
the STALE binary, and a stale binary prints green. M1, M2, M5, M7 and M13 each
did exactly that. They are re-run with the one `(void)x;` or `[[maybe_unused]]`
that silences the warning and changes nothing the mutation is about, and only
the second reading is recorded. This is the third time in this campaign that a
build failure has presented as a pass; reading the build rc first is what caught
it.

| # | mutation | file | verdict |
|---|---|---|---|
| M1 | `hist[t + k * dilation]` → `hist[t + k]`: the taps read at unit stride | kernel | **RED, 5 of 10 cases, 7 of 544 assertions** |
| M2 | `dilation = args.dilation` → `dilation = 1`: the arg is never read | kernel | **RED, 5 of 10 cases, 7 of 544** |
| M3 | the tap order reversed, `weight[c*K + k]` → `weight[c*K + (K-1-k)]` | kernel | **RED, 5 of 10 cases, 9 of 546** |
| M4 | the state write-back one column early, `hist[tokens+j]` → `hist[tokens+j-1]` | kernel | **RED, 4 of 10 cases, 263 of 538** |
| M5 | the silu dropped from the store | kernel | **RED, 5 of 10 cases, 9 of 546** |
| M6 | the state keeps the ACTIVATED value instead of the raw conv input | kernel | **RED, 4 of 10 cases, 263 of 538** |
| M7 | `conv_state_indices` ignored: row `s` for sequence `s` unconditionally | kernel | **RED, 1 of 10 cases, 3 of 538** |
| M8 | the per-sequence token offset dropped on the `x` load | kernel | **RED, 1 of 10 cases, 113 of 538** |
| M9 | the tap accumulator narrowed from `double` to `float` | kernel | **RED, 1 of 10 cases, 1 of 538** — the model-width case, which asserts BIT-IDENTITY with the host reference by `memcmp`. No golden comparison at C = 16 can see this; the 10240-channel agreement check is the only thing that does |
| M10 | the empty-segment early-out removed | kernel | **SURVIVED — and it is an EQUIVALENT MUTANT, not a gate hole.** With `tokens == 0` the span is `state_len`, the window loop does not execute, and the write-back reads `hist[0 + j]`, which is the column it then writes: the two programs compute the same function. The dispatcher refuses a decreasing `query_start_loc`, so `0` is the only value that reaches the branch. It is kept as a PERFORMANCE early-out — at 10240 channels a padded batch row would otherwise cost 184k pointless float copies per layer — and the kernel comment says that in those words, because the comment that stood there first claimed it stopped the cache being shifted and it does not. The repair is M16, which mutates the same territory in a way a test can see |
| M11 | the `(K-1)*dilation` state-width check widened to `>= K-1` | dispatcher | **RED, 1 of 10 cases, 1 of 538** — the Mamba-shaped-state refusal |
| M12 | the `query_start_loc` bounds check removed | dispatcher | **RED, 1 of 10 cases, 1 of 538** |
| M13 | the `conv_state_indices` range check removed | dispatcher | **RED, rc = 134 (SIGABRT), 1 of 10 cases, 1 of 535** — the refusal assertion reports `did NOT throw at all!`, and the unchecked row index (7 into a cache of 3) then writes past the allocation, which glibc catches as `double free or corruption (out)` and turns into `SIGABRT`; doctest prints `FATAL ERROR: test case CRASHED: SIGABRT`. **The SIGNAL is not stable and the row must not be read as if it were.** The first record here said `rc = -6`, which was a negative `WTERMSIG` written where a shell exit status belongs; the fresh reviewer of this wave measured `rc = 139` (`SIGSEGV`, core dumped) on the same case and the same 1-of-10 / 1-of-535 counts; this re-run measured 134. All three are the same defect. An out-of-range row index writes at `row_stride * 7` past a three-row cache, and whether that lands in unmapped memory (`SIGSEGV`) or in allocator bookkeeping the next free checks (`SIGABRT`) is a property of the heap layout, not of the mutation. What is stable, and what the row is actually evidence for, is the assertion count: the refusal is the ONLY thing standing between a caller error and undefined behaviour, which is why it is a check rather than a comment. The re-run deleted the whole `if (conv_state_indices != nullptr)` block, declaration included, so unlike the first pass it needed no `(void)` silencer and built at rc 0 — the build rc was read before the run rc, because a stale binary prints green |
| M15 | the segment loop stops after the first sequence | kernel | **RED, 2 of 10 cases, 114 of 538** |
| M16 | an empty segment RESETS its cache row instead of leaving it | kernel | **RED, 1 of 10 cases, 1 of 538** — M10's repair: the plausible defect in that territory is clobbering a padded row, and the empty-segment case sees it |
| M14 | **REACHABILITY**: the `RegisterOp(OpId::kQwen4ExpPleConv, DeviceType::kCPU, ...)` line deleted | kernel | **RED, 9 of 10 cases, only 9 assertions reached** — every case that calls the op throws `vt: no kernel for op Qwen4ExpPleConv (id 134) on device cpu`. `[[maybe_unused]]` on the kernel is required or `-Werror=unused-function` fails the build and the stale binary reads green |

The M14 shape is the load-bearing reachability proof AVAILABLE AT THIS LAYER,
and it is not the one AGENTS.md `## Nothing lands dead` really wants. Deleting a
production call site is impossible here because there is no production call site
— see `## Owed` — so what M14 proves is that the tests enter the op through the
dispatcher and the registry rather than through the kernel function, which is
the strongest statement this slice can make.

**The RED that came first.** Before the kernel existed, with the OpId, the args
struct, the dispatcher and the test all present, `test_qwen4_exp_ple_device`
reported 8 of 9 cases failing with
`vt: no kernel for op Qwen4ExpPleConv (id 134) on device cpu (type 0)` at
`src/vt/op_provider.cpp:577`. The one case that passed was the refusals case,
whose subcases all throw in the dispatcher before reaching `GetOp` — which is
itself the evidence that the geometry checks are in the dispatcher and not in
the kernel.

## Mutation record — W5b-2 (#2123)

The device arm of the gated-residual stream, `vt::Qwen4ExpGatedResidual` and
`vt::Qwen4ExpGatedResidualWriteBack`. Method as in the two sections above: one
textual change applied to a pristine tree, proved applied by a **sha256**
comparison rather than by `git diff` — the kernel translation unit is NEW on
this branch and an untracked file has an empty diff no matter what is written
into it — then `touch`ed, rebuilt, run, restored from a byte-identical copy,
`sha256sum`-verified, `touch`ed again and rebuilt. Without that second touch
ninja skips the rebuild and the mutations ACCUMULATE, which fails toward RED and
makes a weak gate read strong. Every row was re-measured on the FINAL head, not
at the point in the repair where its fix landed.

**Two mutations survived the first battery, and both are repaired here rather
than recorded and left.** The table below is the second battery; the first
battery's verdict is in the right-hand column where it differs, because a
mutation table whose only rows are reds is a table nobody re-ran.

Target `src/vt/cpu/cpu_qwen4_exp.cpp` unless stated; suite
`tests/vllm/models/test_qwen4_exp_hc_device.cpp`, 9 cases / 87 assertions green.

| # | mutation | result | first battery |
|---|---|---|---|
| M1 | `/ hc_count` moved OUTSIDE the SiLU | RED, 2 of 9 cases, 4 assertions | RED |
| M2 | a `/ hc_count` ADDED to the up-projection sigmoid | RED, 2 of 9, 4 | RED |
| M3 | the branch collapse made a SUM instead of a MEAN | RED, 4 of 9, 7 | RED |
| M4 | the gate multiplied against the RAW stream, not the normed one | RED, 4 of 9, 7 | RED |
| M5 | injection sigmoid loses its `/ hc_count` | RED, 1 of 9, 3 | RED |
| M6 | injection loses its `2 *` | RED, 2 of 9, 5 | RED |
| M7 | `eps` moved OUTSIDE the rsqrt (added to the norm, not the mean square) | **RED, 1 of 9, 2** | **SURVIVED** |
| M8 | grouped norm collapsed to a whole-row norm over `hc*H` | RED, 4 of 9, 10 | RED |
| M9 | the per-group sum of squares accumulated in `float` | **RED, 1 of 9, 1** | **SURVIVED** |
| M10 | write-back reads `block_out` backwards | RED, 2 of 9, 5 | RED |
| M11 | write-back assigns instead of accumulating | RED, 2 of 9, 5 | RED |
| M12 | every token reads token 0's stream (the batch axis dropped) | RED, 4 of 9, 12 | RED |
| M13 | the normed stream rounded through the stream dtype (upstream's `.type_as(x)`) | RED, 1 of 9, 8 | RED |
| M14 | `x / hc` replaced by `x * (1.0f / hc)` | **SURVIVED**, 9/9, 87 | SURVIVED |
| M15 | the write-back op's `RegisterOp` line deleted, **and `[[maybe_unused]]` added to `Qwen4ExpGatedResidualWriteBackKernel`** | RED, 2 of 9, and the suite runs 74 assertions instead of 87 | RED |
| M16 | the SAME eps mutation in the HOST reference `qwen4_exp_hc.cpp`, against `test_qwen4_exp_hc` | **RED, 3 of 15 cases, 10 assertions** (pre-repair, at `origin/main`: RED, 2 of 14, 2 — W3's `big_eps` probe already gated it) | n/a |
| M17 | M16's mutation against the DEVICE suite | SURVIVED, 9/9, 87 | n/a |
| M18 | the host reference's mean collapse made a SUM, against the DEVICE suite | RED, 1 of 9, 2 | n/a |

**What the two repairs were, and why the first battery could not see either.**

*M7, the epsilon placement.* `eps` goes inside the rsqrt, added to the MEAN
SQUARE (`torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`,
`modeling_qwen4_exp.py:170`); the plausible slip adds it to the norm instead.
Golden cases A, B and C all draw the stream at `hyper_scale = 1.7`, where the
mean square is O(1) and `eps = 1e-6` moves the answer by about 5e-7 — a third of
the suite's own `kTol`, so the wrong spelling passes. The same is true of every
GOLDEN case in the W3 host suite, and its `Variant` sweep has no flag for it
either — but **W3 did not ship only goldens.** It shipped a deliberate large-eps
probe (`test_qwen4_exp_hc.cpp:268-276`) that pins the placement against
`NormRefD` at `big_eps = 4.0`, and on a reconstructed pre-repair tree the eps
mutation reds it: `CHECK( 0.802185 < 1e-05 )` at :276, **2 of 14 cases and 2
assertions** over the whole suite. The hole was the DEVICE arm's, which had no
such probe. M16 is in this table because case D raises the host arm's answer from
those 2 assertions to 10, not because the host arm was silent. The repair is a
fourth golden case, `D`, drawn at `hyper_scale = 0.01` — mean square ~1e-4, so
`eps` is 1% of it and the
two spellings separate by ~0.5%, three orders over `kTol`. `scripts/gen-qwen4-exp-hc-goldens.py`
grew one `hyper_scale` parameter, defaulted to the existing 1.7, and A/B/C
regenerate byte-identically (verified: the only diff in those three blocks is the
comment line that now prints the scale). Case D is driven by BOTH suites.

*M9, the accumulator width.* The per-group sum of squares runs over 2560 terms in
the real model and `double ss` -> `float ss` is a one-word edit that changes
nothing measurable on ordinary data. The W3 suite already gates it — at the real
group size, on magnitude-separated data — but it gates the HOST reference, not
this kernel, and the device suite's own real-width case draws uniform inputs where
the two accumulators agree. The repair is a device case built on the same
principle and with one addition: `mix_down` and `mix_up` are ZEROED, so `silu(0)`
is exactly 0, every gate is `sigmoid(0) = 0.5` exactly, and `mixed` is a four-term
f32 mean of the normed stream with the reduction under test as its ONLY remaining
error source. Without that the two f32 projections over `flat = 10240` contribute
their own ~1e-6 and sit on top of the signal. Measured on exactly that data:

| | value |
|---|---|
| max abs of the double reference | 3.2895e+01 |
| ours, `double` accumulator | 1.173e-06 (3.6e-08 relative, ~0.6 ulp) |
| the same kernel with `float ss` | 6.702e-04 (**571x worse**) |

The bound is 1e-5: 8.5x above the first and 67x below the second.

**M15's recipe needs the attribute, and without it the row is unmeasurable.**
Deleting the `RegisterOp` line alone leaves the kernel defined and uncalled, and
this tree builds with `-Werror=unused-function`, so the translation unit does not
compile: `error: 'Qwen4ExpGatedResidualWriteBackKernel' defined but not used`.
The build then exits non-zero, the STALE device binary is still on disk, and
running it prints 9 / 87 SUCCESS — the false green the W5b-1 section above names
in the same words. Read the build RC first, and add `[[maybe_unused]]` to the
kernel's declaration so the deletion under test is the registration and nothing
else. With the attribute the recorded result reproduces exactly: build RC 0,
2 of 9 cases red on `no kernel for op Qwen4ExpGatedResidualWriteBack`, 74
assertions instead of 87.

**M14 is a real survivor and it stays one.** `x / hc` and `x * (1.0f / hc)` differ
by at most one ulp for an `hc` that is not a power of two, which is golden case B
at `hc_count = 3`; the difference propagates through a SiLU and a projection and
arrives at `mixed` some four orders below `kTol`. No comparison against f32
goldens can separate them, and tightening `kTol` far enough would fail the
unmutated kernel first (the measured unmutated max abs difference over case B is
2.384e-07). The correct spelling is kept because upstream spells it `/
self.hc_count` and the host reference already argues the point, not because a gate
proves it. Recorded so nobody reads its absence from the red list as coverage.

**M17 is the honest shape of an agreement check.** It applies M16's epsilon
mutation to the host reference and runs the DEVICE suite, which reaches that
reference only through its two model-width cases — where `eps = 1e-6` against a
mean square of ~0.33 is invisible, exactly as it is in cases A/B/C. It says the
model-width comparison cannot see an epsilon defect, and M18 is the row that says
the comparison is nonetheless LIVE: a sum-for-mean change in the same reference
reds it. Both rows are here because a single survivor with no companion reads as
an instrument that is not wired up.

## Mutation record — W5b-4 (#2167)

Qwen Sparse Attention on the device arm: `vt::Qwen4ExpQsaCompress`,
`vt::Qwen4ExpQsaGatherAttention`, and the indexer COMPOSED from
`vt::DsaIndexerLogits` + `vt::DsaTopkSelect` (see `### W5b-4 correction` above
for why those two are not re-implemented here). Method as in the sections above:
one textual change applied to a pristine tree, proved applied by a **sha256**
comparison rather than by `git diff` — both new files are UNTRACKED on this
branch and an untracked file has an empty diff no matter what is written into it
— then rebuilt, run, restored from a byte-identical copy and `sha256sum`-verified
against the pre-mutation digest. **The build return code is read BEFORE any test
result**: the first battery had three mutations that failed to compile under
`-Werror` (`half`, `groups` and `prev` become unused when the line that reads
them is replaced), and a `ninja` failure leaves the previous binary in place, so
each of the three would otherwise have run a STALE binary and reported a pass.

Every row below is re-measured on the FINAL head, after the repair, not at the
point in the wave where its fix landed. Target `src/vt/cpu/cpu_qwen4_exp_qsa.cpp`
unless stated; suite `tests/vllm/models/test_qwen4_exp_qsa_device.cpp`,
**12 cases / 4697 assertions green** after the fresh review's repair added the
unmapped-tail probe. The per-mutation figures in the table below were measured on
the 11-case suite, before that case existed; the three rows the repair added or
re-measured (the probe row, M11 and M11c) name the 12-case suite explicitly.

| # | mutation | build rc | result | first battery |
|---|---|---|---|---|
| M1 | pool stores a SUM, not a mean (drop the / compress_ratio) | 0 | RED, 2 of 11 cases, 9 assertions | RED |
| M2 | eps OUTSIDE the rsqrt instead of added to the mean square | 0 | RED, 2 of 11 cases, 9 assertions | RED |
| M3 | RoPE at the block's LAST position instead of its first | 0 | RED, 4 of 11 cases, 53 assertions | RED |
| M4 | vLLM norm polarity `* w` instead of upstream's `* (1 + w)` | 0 | RED, 6 of 11 cases, 1516 assertions | RED |
| M5 | rotate_half loses its minus sign | 0 | RED, 3 of 11 cases, 39 assertions | RED |
| M6 | GPT-J adjacent-pair rotation (DeepSeek-V4's) instead of NeoX half-split | 0 | RED, 3 of 11 cases, 35 assertions | build failed |
| M7 | the pool's `.to(dtype)` bf16 round-trip dropped | 0 | RED, 1 of 11 cases, 2 assertions | RED |
| M8 | OVERLAPPING pooling window (DeepSeek-V4's shape), stride 1 not CR | 0 | RED, 5 of 11 cases, 1548 assertions | RED |
| M9 | the RMS norm's `.type_as(x)` bf16 rounding dropped | 0 | RED, 1 of 11 cases, 2 assertions | RED |
| M10 | the NoPE dims are zeroed instead of carried through untouched | 0 | RED, 5 of 11 cases, 1479 assertions | RED |
| M11 | THE LOAD-BEARING ONE: a dense masked walk reporting the SPARSE read count | 0 | RED, 1 of 11 cases, 256 assertions; **re-measured on the 12-case suite: RED, 2 of 12 cases, 257 assertions** — the NaN case and the unmapped-tail probe | **SURVIVED** |
| M11b | the same dense masked walk, with the counter left AT the read site | 0 | RED, 3 of 11 cases, 260 assertions | n/a |
| M11c | THE FETCH, NOT THE MULTIPLY: prefetch every cached row and discard it, then gather honestly | 0 | RED, **1 of 12** cases — the unmapped-tail probe ALONE; the NaN case and every read-count case pass | n/a |
| M12 | the always-attended ragged tail dropped | 0 | RED, 5 of 11 cases, 899 assertions | RED |
| M13 | block b expands to tokens [b, b + CR) instead of [CR*b, CR*b + CR) | 0 | RED, 4 of 11 cases, 1538 assertions | RED |
| M14 | block b expands to CR - 1 tokens (the off-by-one) | 0 | RED, 5 of 11 cases, 1926 assertions | RED |
| M15 | GQA head mapping `h % HKV` instead of `h / groups` | 0 | RED, 4 of 11 cases, 1154 assertions | build failed |
| M16 | the softmax scale dropped from the logit | 0 | RED, 3 of 11 cases, 1907 assertions | RED |
| M17 | reads counted once per row instead of once per pass | 0 | RED, 2 of 11 cases, 4 assertions | RED |
| M18 | the gather visits its rows in DESCENDING order | 0 | RED, 3 of 11 cases, 1577 assertions | RED |
| M19 | the ASCENDING/in-range block refusal deleted | 0 | RED, 1 of 11 cases, 1 assertions | build failed |
| M20 | the COMPLETE-blocks refusal deleted from the compressor dispatcher (`src/vt/ops.cpp`) | 0 | RED, 1 of 11 cases, 1 assertions | RED |
| M21 | the rotary-fits-the-index-head refusal deleted (`src/vt/ops.cpp`) | 0 | RED, 1 of 11 cases, 1 assertions | RED |
| M22 | the explicit-scale refusal deleted from the gather dispatcher (`src/vt/ops.cpp`) | 0 | RED, 1 of 11 cases, 1 assertions | not applied |
| M23 | the cos/sin coverage refusal deleted (`src/vt/ops.cpp`) | 0 | RED, 1 of 11 cases, 1 assertions | RED |
| M24 | REACHABILITY: the compressor's RegisterOp deleted | 0 | RED, 10 of 11 cases, 0 assertions | RED |
| M25 | REACHABILITY: the gather's RegisterOp deleted | 0 | RED, 6 of 11 cases, 1 assertions | RED |
| M26 | CONTROL: inherit DeepSeek-V4's n_head_scale, which QSA has no tensor for (the TEST) | 0 | **SURVIVED**, 11/11, 4427 | **SURVIVED** |
| M27 | the indexer's `weights` stop being ones, so the fold no longer collapses (the TEST) | 0 | RED, 3 of 11 cases, 1269 assertions | RED |
| M28 | the scoring window becomes the whole cache, not the visible complete blocks (the TEST) | 0 | RED, 4 of 11 cases, 0 assertions | RED |

**THE SURVIVOR, AND ITS REPAIR.** M11 is the defect this whole wave exists to
prevent, and in the first battery it SURVIVED: 10 of 10 cases, 4167 of 4167
assertions, against a body that walked every one of the `kv_len` cached rows with
a `-inf` mask and reported `sel.size() * 2` per head as its read count. That is
W4's M22c reproduced one layer up, and the reason is structural rather than
careless. A `keys_visited` the kernel writes cannot convict the kernel that
writes it, whatever the counter's placement in the SHIPPED code, because a
mask-shaped port changes the loop and the counter together — that is what a
mask-shaped port IS. And no value comparison can convict it either: `exp(-inf -
m)` is exactly +0 and adding an exact zero changes no accumulator, so a mask
agrees with a gather value for value, which is precisely why a token gate lets
one through.

The repair is an observable of the WALK rather than of the bookkeeping. The case
`vt::Qwen4ExpQsaGatherAttention: the unselected rows are NaN and the answer is
finite` poisons every cached row the selection does not name with `NaN`, in both
K and V, for a single query token whose complement is therefore well defined
(23 cached, 11 attended, 12 poisoned). A gather never addresses those rows and is
bit-identical to the same gather over a clean cache. A mask reads every value row
and accumulates `w * v` with `w == 0.0f`, and `0.0f * NaN` is `NaN` in IEEE-754,
so its output is `NaN` in every lane. M11 reds on it.

M11b is the companion that says the counter is nonetheless live: the SAME dense
masked walk with `++reads` left at the read site reds the read-count cases
directly, so the shipped counter is a function of the walk and not a restatement
of the selection. A single survivor with no companion reads as an instrument
nobody wired up, which is why both rows are here.

**THE LIMIT OF THE NaN PROBE, AND THE PROBE THAT CLOSES IT.** The NaN poison
proves a row was not multiplied into the accumulator. It does not prove the row's
bytes were never fetched: a body that loads every row and discards the unselected
ones before the multiply passes it, and the loop counter is not the cost
llama.cpp #27739 measures anyway — the key-row traffic is.

The wave's first draft recorded that gap as blocked, on the reasoning that the
structural version is a PAGED cache whose unselected blocks are not mapped, and
that the block-table store is owed and waits on
[#2131](https://github.com/mudler/vllm.cpp/issues/2131). **That is true of the
PRODUCTION cache and false of a test instrument**, and the fresh review proved it
by building one. `vt::Qwen4ExpQsaGatherAttention: the gather never FETCHES an
unmapped unselected row` `mmap`s its own page-aligned K and V caches at kv_len
3000 — a multiple of `compress_ratio`, so the always-attended ragged tail is
empty — selects blocks `0..511`, and `mprotect(PROT_NONE)`s the 59 whole pages
(241664 bytes, `[524288, 768000)`) that lie strictly inside the unselected run
`[2048, 3000)`, in BOTH caches. The shipped kernel walks past the hole:
`keys_visited` 16384 against a dense 24000, and its output is bit-identical to
the same call over the unguarded mapping. M11's dense masked walk dereferences
the first guarded row and takes SIGSEGV.

The construction is forced by the kernel, not chosen: the gather addresses the
cache as `(p * HKV + kvh) * DH + d` and never reads `key.stride[0]`, so a guard
page BETWEEN rows is unavailable and the unselected rows have to form one
contiguous tail.

**M11c is what says the probe is not a restatement of the NaN case.** It
prefetches every cached row into a discarded accumulator and then gathers
honestly with the honest counter — the exact body the paragraph above names as
the NaN probe's blind spot. It reds **one** case out of twelve, the unmapped-tail
probe, and passes the NaN case, every read-count case and every value case. The
two probes are therefore ordered, not redundant: NaN convicts the multiply, the
unmapped tail convicts the fetch.

**A fault has to be a failing assertion, not a dead binary.** doctest installs a
fatal-condition handler around every case, and left in place it turns the
mutant's SIGSEGV into `FATAL ERROR: test case CRASHED` and abandons the rest of
the run — every remaining case reads as skipped, so one convicted mutant costs
the verdict on every other property in the file. This was measured, not assumed:
the first build of the probe omitted the handler and exited 139 with eleven cases
unreported. The probe installs its own `SIGSEGV`/`SIGBUS` handler around the one
call, `siglongjmp`s back into the case, restores doctest's handlers, and reports
`CHECK_FALSE(faulted)`. On a platform without POSIX `mmap`/`mprotect` the case is
declared `doctest::skip()` rather than compiled out, because a probe that cannot
run must say so.

**THE OTHER SURVIVOR IS A DELIBERATE CONTROL.** M26 inherits DeepSeek-V4's
`n_head_scale = n_head ** -0.5` into the composed indexer, which QSA has no
tensor for, and the suite stays green. That is not a hole: top-k is invariant
under a positive rescale of every score, so the mutation cannot move a selection
by construction — the same blindness W4 already recorded for `QsaBlockScores`,
and the reason that constant is gated by a hand-derived VALUE case in the host
suite rather than by any selection. M27 is its paired red: making the indexer's
`weights` non-uniform breaks the fold's collapse to a single constant and reds
3 of 11 cases, so the composition's `weights == 1` is load-bearing rather than
decorative.

**WHAT THE CONTEXT LENGTH BUYS, measured.** M12 (the ragged tail dropped) and
M14 (a block expanded to `CR - 1` tokens) both red, and both would red at the
golden shapes alone. The case that only the released-config context can carry is
the sparsity itself: at kv_len 2051 — the sub-budget control — `keys_visited`
equals the dense figure exactly, so every read-count assertion in this file is
trivially true there and a mask passes them all. At kv_len 3002 the gather reads
2050 of 3002 rows per query token and the same assertions bite. A QSA gate that
never crosses 2048 is not a weaker gate; it is not a gate.

## Mutation record — W5b-5 (#2211)

`Qwen4ExpTextAttention` as one production block, and the first place under
`src/` that COMPOSES the QSA indexer. Method as in the sections above: one
textual change applied to a pristine tree, proved applied by a **sha256 that
moved** (the file is NEW on this branch and an untracked file has an empty diff
whatever is written into it), the file `touch`ed so ninja cannot skip the
rebuild, the **BUILD RETURN CODE READ BEFORE ANY TEST RESULT**, then restored
from a byte-identical copy and `sha256sum`-verified against the pre-mutation
digest `ab132cafcd327dda…`. Every row was re-measured on the FINAL head, after
the repair. Target `src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp`;
suite `tests/vllm/models/test_qwen4_exp_qsa_block.cpp`, **8 cases / 2828
assertions green** at the time this table was measured; the fresh-review repair
recorded below took that to **8 / 2831**.

**THE RED CAME FIRST, AND IT WAS MEASURED RATHER THAN ASSERTED.** With the suite
and the header in place and both function BODIES replaced by a refusal, build
rc 0, the run reported **7 of 7 cases failing, 8 of 11 assertions**, every one on
`qwen4_exp qsa block: not composed yet`. Nothing in this file has ever passed
against an absent composition.

**Two mutations failed to BUILD on the first pass, which is the fourth time in
this campaign that a build failure has presented as a pass.** M3 leaves `one`
unused and M4 leaves `kl` unused; this tree builds with `-Werror`, ninja then
leaves the STALE binary on disk, and a stale binary prints green. Both are re-run
with the one `(void)x;` or `[[maybe_unused]]` that silences the warning and
changes nothing the mutation is about, and only the second reading is recorded.
Reading the build rc before the run rc is what caught it.

| # | mutation | build rc | result |
|---|---|---|---|
| M1 | **setting 2**: inherit DeepSeek-V4's `n_head_scale = n_head ** -0.5`, which QSA has no tensor for | 0 | **RED, 1 of 8 cases, 2 assertions** — the LOGITS VALUE case alone, which is the whole point of it. This is the repair for M26, which survived at the op layer because no selection can move under a positive rescale |
| M2 | **setting 3**: the softmax scale dropped from the fold | 0 | **RED, 1 of 8, 2** — again the value case alone |
| M3 | **setting 1**: the indexer `weights` stop being ones, so the per-head fold no longer collapses to one constant | 0 | RED, 3 of 8, 7 |
| M4 | **setting 4**: the scoring window becomes the WHOLE cache instead of the visible complete blocks | 0 | RED, 5 of 8 — by REFUSAL, not by assertion: a query then selects a block it cannot see and the gather's own in-range check throws. 113 assertions reached |
| M5 | setting 4 at the other end: `win_start` skips the first block | 0 | RED, 6 of 8, 2 assertions of 113 — the value case reds, then the gather refuses a query that attends nothing |
| M6 | the pooled key's rope span derived from the INDEXER head width instead of the model's | 0 | RED, 8 of 8, 11 assertions reached — a REFUSAL mutation: it trips the compressor's cos/sin coverage check |
| M6b | the pooled key is never roped at all (`rotary_dim = 0`), which is IN RANGE | 0 | RED, 8 of 8 — the arithmetic companion to M6, so the territory is not gated by a refusal alone |
| M7 | the pooled key's bf16 round-trip dropped | 0 | RED, 1 of 8, 1 |
| M8 | the ragged tail rounded UP into a block that does not exist yet | 0 | RED, 8 of 8, 11 reached — a REFUSAL mutation: this file's own `RowsView` range check |
| M8b | the LAST complete block dropped, which is IN RANGE | 0 | RED, 5 of 8, 4 assertions — the arithmetic companion to M8 |
| M9 | the indexer q norm applies `out * w` instead of upstream's `out * (1 + w)` | 0 | RED, 1 of 8, 1 |
| M10 | the model q norm loses the `+1` fold | 0 | RED, 2 of 8, 3 |
| M11 | the model k norm loses the `+1` fold | 0 | RED, 2 of 8, 3 |
| M12 | the side cache stores a NORMED indexer key, which the compressor then norms again | 0 | RED, 4 of 8, 38 |
| M13 | the indexer key ignores `past_len` and overwrites row 0 | 0 | **RED, 1 of 8, 2 — after the repair. It SURVIVED the first battery**, see below |
| M14 | the attention softmax scale taken from the INDEXER head dim | 0 | RED, 2 of 8, 3 |
| M15 | the query and the OUTPUT GATE halves of `q_proj` swapped | 0 | RED, 2 of 8, 3 |
| M16 | the sigmoid OUTPUT GATE dropped entirely | 0 | RED, 2 of 8, 3 |
| M17 | the key is never roped, only the query | 0 | RED, 2 of 8, 3 |
| M18 | GPT-J adjacent-pair rotation (DeepSeek-V4's) instead of NeoX half-split | 0 | RED, 3 of 8, 131 |
| M19 | the value cache ignores `past_len` | 0 | RED, 1 of 8, 1 |
| M20 | **THE LOAD-BEARING ONE**: the consumer is handed EVERY VISIBLE block — a dense walk wearing a gather's clothes | 0 | **RED, 3 of 8, 130 assertions** — the golden case at over-budget, the NaN-poison case and the released-config sparsity case |
| M21 | the key is roped into a SCRATCH, so the cache keeps the un-roped key | 0 | RED, 2 of 8, 3 |
| M22 | **the rope cross-check's production call site deleted** — the fresh-review repair below | 0, after `[[maybe_unused]]` silences the `-Werror=unused-function` this leaves | **RED, 1 of 8, 2** — both new refusal subcases stop throwing |

**THE SURVIVOR, AND ITS REPAIR.** M13 writes the indexer key at cache row 0
instead of row `past_len`, and in the first battery it SURVIVED: 8 of 8 cases,
2805 of 2805 assertions. The reason is structural rather than careless. Every
golden case is a PREFILL, where `past_len` is 0 and the two spellings are the
same expression; the one case with `past_len > 0` compared the block OUTPUT at a
bf16-sized relative bound, and one wrong pooled key moves the selection by one
block out of five and leaves the answer inside it. The repair is an observable of
WHERE the write landed rather than of what it was worth: the decode case now
compares the side cache ROW FOR ROW against `k...IdxKRaw`, the oracle's own raw
indexer keys, after a split prefill/decode. M13 reds on it.

**WHAT THE VALUE GATE MEASURED, and why its bound is 1e-6 rather than 1e-3.**
Fed `k...IdxQPost` and `k...IdxKRaw` — the oracle's own roped indexer query and
raw keys, captured by intercepting `apply_rotary_pos_emb` and slicing
`index_qk_proj`'s output — the composed logits are BIT-IDENTICAL to the oracle's
own pre-top-k `scores`: max abs **0** over a scale of 3.365 (12 logits,
sub-budget) and 6.239 (60 logits, over-budget). The reassociation the spec names
(upstream divides AFTER the head sum, the op's fold multiplies BEFORE it) does
not move a value at these shapes. Fed this port's OWN bf16 projection and bf16
RoPE instead, the same comparison lands at 1.76e-3 and 1.50e-3 — measured, not
feared, and it is why the fixture captures the oracle's inputs at all. The two
are separate cases: one gates the CONSTANTS, the other gates the INPUTS
(measured 0 and 3.8e-3 / 4.2e-3 against a bf16-sized 1e-2), and neither pretends
to be the other.

**THE OTHER MEASURED MARGINS**, so that a later reader can see which bound is
load-bearing and which is slack. Block output vs the oracle: 5.05e-3 and 5.58e-3
against 3e-2 (5.4x). The decode step: 6.13e-3 against 3e-2. The roped indexer
query: 4.22e-3 against 1e-2, a 2.4x margin and the tightest in the file — one
bf16 ulp is the floor there, because `vt::RopeFromCache` computes each rotated
pair in f32 and stores once where upstream multiplies and adds in bf16.

**WHAT THE CONTEXT LENGTH BUYS, measured again at the BLOCK layer.** At kv_len
3002 with the released indexer values (budget 2048, compress ratio 4) the block
reads `keys_visited` **16400** against a dense **24016** — 512 blocks x 4 rows
plus the 2-token ragged tail, x 4 query heads x 2 softmax passes. Below 2048
every candidate is selected and that assertion is trivially true, which is why
the case exists at 3002 and not at the fixture's 23.

**THE NaN PROBE AT THE BLOCK LAYER, and what it does and does not add.** The
block is run twice over identical inputs, the second time over a cache whose 12
of 23 rows the block's OWN selection does not name are bf16 NaN in both K and V
(11 attended). The output is finite AND bit-identical to the clean run, and
`keys_visited` agrees. That convicts a mask at the BLOCK's consumer call, which
the op-layer probe cannot do because it does not know what the block selects.
What it does NOT add is the fetch-level property — that the bytes were never
READ — which stays discharged by the `mprotect(PROT_NONE)` probe in
`test_qwen4_exp_qsa_device.cpp`; M20 is what says this block's only consumer call
is the op that probe covers.

**THE FRESH-REVIEW REPAIR: A HEADER THAT CLAIMED A GATE NOBODY HAD WRITTEN.**
The header said of the two rope layouts "the caller builds both from one table
and the gate asserts they agree". Nothing asserted it. The test's own `BuildRope`
DERIVES `packed` from `cos`/`sin`, so the two agree BY CONSTRUCTION in every case
in the file — which is not an assertion, and a layer loop that built them
inconsistently would diverge silently while the header told the next reader it
was covered. The block accepted two independently supplied layouts and
cross-checked neither.

Repaired by writing the check rather than softening the sentence, because a
silent divergence in a layer loop is the failure this campaign keeps finding.
`CheckRopeLayoutsAgree` refuses unequal heights, then compares a BOUNDED SAMPLE
of rows — row 1, the midpoint and the last row — value for value at one bf16 ulp
(2^-7); it refuses a non-CPU-resident pair BY NAME rather than skipping, because
a check that silently does not run on a device arm is a mute switch. Row 0 is not
a probe: cos is 1 and sin is 0 at every frequency there, so it agrees under every
difference the probe exists to catch. A full comparison is not paid, and that is
a cost decision rather than an oversight: it would be O(P * rot) per QSA layer per
step to re-check a constant.

**THE RED CAME FIRST, AND IT MEASURED THE SILENCE.** Two subcases were added
BEFORE the check existed. One perturbs the FULL `cos` table at row `c.seq - 1`
— a row nothing else in the block reads, because `vt::RopeFromCache` reads the
PACKED cache and the compressor reads only block-start rows (multiples of 4, and
10 is not one at seq 11); the other hands the two layouts different heights. On
the pre-check head, build rc 0, both reported **did NOT throw at all** and the
other **2829 of 2831 assertions passed** — so the divergence was invisible to
every value gate in the file, which is the claim the header had been making in
reverse. With the check in place: **8 of 8 cases, 2831 of 2831 assertions**. M22
above deletes the production call site on the repaired head and reds both
subcases again, which is what says the subcases gate the CHECK and not the
arithmetic.

**M1 AND M13 WERE RE-ARMED ON THE REPAIRED HEAD**, because a later commit can
silently disarm an earlier commit's mutation proof. The table above was measured
against the pre-repair file (digest `ab132cafcd327dda…`); the repaired file is
`e837cf290a86bd0d…`. M1 (inherit `n_head ** -0.5`) reds 1 of 8 / 2 assertions and
M13 (the indexer key at row 0) reds 1 of 8 / 2 assertions, both unchanged from
their recorded rows. The remaining rows are not re-measured, and the reason is
stated rather than assumed: the added check reads ONLY the two rope operand
tables and compares them with each other, and no mutation in the table alters
what a caller passes in those two arguments.

**NO REACHABILITY MUTATION IS AVAILABLE AT THIS LAYER, and that is the honest
statement rather than an omission.** AGENTS.md `## Nothing lands dead` wants a
production call site deleted, and there is none to delete: `## Owed` records this
block as unreached with the row and the issue that own the wiring. M20 is the
strongest statement this slice can make — it proves the tests enter the consumer
through `vt::Qwen4ExpQsaGatherAttention` and not through some other path.
## Mutation record — W5c-1 (#2031)

Every mutation was sha256-proven applied, **its BUILD rc was read before any
test result**, the tree was restored byte-for-byte with the hash re-checked, and
the final head was re-measured green afterwards. `runner.cpp` was measured at
`e538172d207f…`, `qwen4_exp_registry.cpp` at `2c30140e7b65…`,
`qwen4_exp_gguf_weights.cpp` at `b88f6e9ba247…`, and all three hashes are the
head's. I is the one row whose CASE gained assertions after its first run (the
`KVBytesPerBlock` pair), so it was re-run on the final head and reads the same
2 cases / 5 assertions.

### The RED, before the change

`test_qwen4_exp_kv_cache` at the branch base, every case entering through
`reg.factory->make_kv_cache`:

```
test_qwen4_exp_kv_cache.cpp:131: ERROR: test case THREW exception:
  Qwen4ExpForConditionalGeneration: the KV-cache spec is not ported yet
  (W4 owes the QSA indexer side cache and W2 the third conv state for the
   n-gram token history). See .agents/specs/qwen4-exp-flash-next.md and #1978.
[doctest] test cases:  3 |  0 passed | 3 failed | 0 skipped
[doctest] assertions: 32 | 23 passed | 9 failed |
```

Green at the head: **4 cases / 399 assertions / rc 0** (the fourth case, the
`--kv-cache-dtype fp8` consequence, was written after the first red).

### Counts, before and after, on the same tree

| Suite | Before | After |
|---|---|---|
| `test_runner` | 31 / 884 / rc 0 | 32 / 990 / rc 0 |
| `test_qwen4_exp_kv_cache` | did not exist | 4 / 399 / rc 0 |
| `test_qwen4_exp_gguf_weights` | 11 / 2970 / rc 0 | 11 / 2975 / rc 0 (one new SUBCASE inside an existing case) |
| `test_qwen4_exp_scaffold` | 12 / 296 / rc 0 | 12 / 296 / rc 0 |
| `test_qwen4_exp_qsa` | 14 / 7263 / rc 0 | 14 / 7263 / rc 0 |
| `test_qwen27_paged_forward` | 31 / 770 / rc 0 | 31 / 770 / rc 0 |
| `test_nemotron_h_paged_forward` | 13 / 3269 / rc 0 | 13 / 3269 / rc 0 |
| `test_kimi_linear_paged` | 8 / 206 / rc 0 | 8 / 206 / rc 0 |

`test_runner` moves by exactly the one case this wave adds. `test_qwen4_exp_qsa`
is byte-identical although its header changed, which is the check that the
#2198 fix touched only comments.

### The battery

| # | Mutation | Build | Result |
|---|---|---|---|
| A | delete `alloc_recurrent_layer_states` **inside `if (multi_cache_topology)`**, in its `membership_by_name && has_mamba_group` recurrent loop | rc 0 | `test_runner` RED — and ONLY the new case, confirmed scoped: `1 case / 0 passed / 1 failed / 31 skipped`, at `REQUIRE(runner.gdn_state().size() == 3)`. The three model suites stay byte-identically green. **This is the `## Owed` item `.agents/specs/recurrent-multistate.md` recorded: at that row's head the same deletion left ALL FOUR suites fully green** |
| B | **CONTROL** — delete the LEGACY single-topology `is_gdn` call site | rc 0 | `test_runner` rc 139 (10 of 13 reached cases failed), `test_nemotron_h_paged_forward` rc 139 (5 of 5 reached), `test_kimi_linear_paged` rc 1 (2 of 8), `test_qwen27_paged_forward` 31 / 770 / rc 0. The deletion harness is LIVE, so A's scoped red is a finding and not a dead instrument |
| C | the recurrent alloc AND view read `state_dtypes[i < 2 ? i : 0]` — states 2 and 3 get `dtypes[0]` | rc 0 | `test_runner` RED, 2 cases. Scoped to the new case: 13 of 106 assertions, every one on `states[3].dtype`, `states[3].Bytes()`, or a total that sums it. The three model suites stay green |
| D | the recurrent view reads `state_shapes[i == 2 ? 1 : i]` — state 2 gets the TEMPORAL shape | rc 0 | `test_runner` RED, 2 cases. Scoped: 16 of 106, on `states[2].rank`, its shape, its bytes and the two byte-identity totals. The three model suites stay green |
| E | publish group 2 as a `FullAttentionSpec` instead of an `MLAAttentionSpec` | rc 0 | `test_qwen4_exp_kv_cache` RED, 2 of 4 cases: the `kMlaAttention` kind, the `MLAAttentionSpec` downcast, and BOTH `fp8` refusal assertions — because a non-MLA third group is one an fp8 cache would silently accept |
| F | delete the `block_size % compress_ratio` refusal | rc 0 | `test_qwen4_exp_kv_cache` RED, 4 assertions, all in the refusal case. Nothing else moves |
| G | delete the non-uniform `attention.compress_ratios` refusal in `Qwen4ExpHfConfigFromGguf` | rc 0 | `test_qwen4_exp_gguf_weights` RED, 2 assertions. **Before this wave the same deletion left that suite fully green** — the refusal existed and gated nothing |
| H | **REACHABILITY** — unhook `.make_kv_cache` from `kQwen4ExpFactory` | **rc 1** | **A BUILD REFUSAL, not a test verdict, and it is read as such:** `error: 'MakeQwen4ExpKVCache' defined but not used [-Werror=unused-function]`. The production factory table is the function's ONLY reference in the tree, so the compiler proves the reach that a test result would only have suggested. No suite ran under this mutation |
| I | drop the `number_of_conv_states() == 3` branch, so the group always publishes two states | rc 0 | `test_qwen4_exp_kv_cache` RED, 2 of 4 cases: the four-shape `REQUIRE`, the 184336 B surcharge, the 51614080 B slack and the 3391504 B page. The uniform-cost accounting is load-bearing rather than decorative |

**Why A needed a NEW fixture and the existing one could not do it.**
`test_runner.cpp`'s "a multi-cache topology keeps its recurrent group" already
combines a multi-cache attention set with a mamba group, and it survives A
untouched: everything it asserts — `layer_kv_class_`, `gdn_group_id_`,
`recurrent_group_ids_`, the per-layer index lists — is computed BEFORE the
allocation loop runs. Classification and allocation are different failures, and
only the second one is what a short KV cache is.

**What the battery did NOT reach**, stated because a battery's silence is not a
result: the four-state group is never allocated on a DEVICE (the CPU host takes
`CacheBuffer`'s host-vector arm), nothing decodes through the published caches,
and no mutation here can see the zero-seeded n-gram history, because no test in
this tree reads that row's CONTENTS. All three are under `## Owed`.

## Mutation record — W5b-6 (#2218)

The gamma-polarity wave. Every mutation was sha256-proven applied, **its BUILD
rc was read before any test result**, the tree was restored byte-for-byte with
the hash re-checked, and both were RE-ARMED on the final head after the registry
comment landed. `cpu_qwen4_exp.cpp` was measured at `4accd54e82be…` and
`qwen4_exp_weights.cpp` at `81328de99cc1…`; both are the head's.

### The RED, before the change

`test_qwen4_exp_forward`, the new composition case, driven through
`ModelRegistry::Load` on the synthetic `qwen4exp` file:

```
tests/vllm/models/test_qwen4_exp_forward.cpp:222: ERROR:
  CHECK( MaxAbsDiff(mixed, want_mixed) < 1e-5f ) is NOT correct!
  values: CHECK( 1.50578 <  1e-05 )
  logged: site layer0.attn_hc
  ... identically at site layer0.mlp_hc and site model.mixer
[doctest] test cases:   1 |   0 passed | 1 failed | 0 skipped
[doctest] assertions: 409 | 406 passed | 3 failed |
```

1.50578 against a 1e-5 bound is not a tolerance question. `w_hf` is in [0, 1) on
this fixture and `1 + w_hf` in [1, 2), so the two parameterizations are a whole
multiplicative unit apart; on the RELEASED checkpoint `w_hf` sits within an ulp
or two of zero and the wrong one produces a stream scaled by ~0.

### Counts, before and after, on the same tree

The base was measured by checking `HEAD~1`'s copies of the four changed files
into this worktree, rebuilding (rc 0) and running, then restoring — not by
quoting the numbers a previous wave recorded.

| Suite | Before | After |
|---|---|---|
| `test_qwen4_exp_forward` | did not exist | 1 / 421 / rc 0 |
| `test_qwen4_exp_hc_device` | 9 / 87 / rc 0 | 9 / 87 / rc 0 |
| `test_qwen4_exp_hc` | 15 / 246 / rc 0 | 15 / 246 / rc 0 |
| `test_qwen4_exp_gguf_weights` | 11 / 2975 / rc 0 | 11 / 2975 / rc 0 |
| `test_qwen4_exp_ple_device` | 10 / 538 / rc 0 | 10 / 538 / rc 0 |
| `test_qwen4_exp_qsa_device` | 12 / 4697 / rc 0 | 12 / 4697 / rc 0 |

The op's numeric contract changed and **every existing count is identical**,
which is the check that the change is a re-parameterization and not a new
answer: the goldens store `w_hf` either way, the fold simply moved from the test
harness into the kernel. The fixture extraction is likewise count-neutral on the
loader suite, 11 / 2975 before and after.

### The battery

| # | Mutation | Build | Result |
|---|---|---|---|
| M-P1 | the kernel drops the `1 +`, i.e. the pre-#2218 contract restored | rc 0 | `test_qwen4_exp_forward` RED 1/1, at all three hyper-connection sites; `test_qwen4_exp_hc_device` RED 4/9, 12 of 87 assertions. The op half is gated |
| M-P2 | `LoadGatedResidual` stops unshifting, `unshift=false` | rc 0 | `test_qwen4_exp_forward` RED 1/1 **at its precondition**, after 6 assertions — the `model_gamma + 1 == file_gamma` `REQUIRE` fires before any arithmetic runs; `test_qwen4_exp_gguf_weights` RED 1/11, 25 assertions. The LOADER half is gated, so a future edit that moves the fold back into `load_weights` cannot land silently |

M-P2 is the half that matters. A case that only reddened on M-P1 would gate the
op against a number the test chose; reddening on both is what makes it a gate on
the SEAM.

### What the battery did NOT reach

Stated because a battery's silence is not a result.

- **The injection arm does not discriminate polarity at this fixture and the
  case says so out loud.** `2 * sigmoid(inject . normed / hc)` runs the
  fixture's `inject` ramp against a 128-wide normed row and reaches ~10^4 under
  BOTH gammas, so the sigmoid saturates at 2.0 either way. The case asserts the
  saturation explicitly, so the day it stops being saturated is loud rather than
  silent, and `mixed` carries the whole discriminating claim.
- **Nothing here decodes.** The composition gated is loader -> one op. The layer
  loop that would put 97 of these calls in sequence does not exist, so no token,
  no `hyper` stream and no `lm_head` is involved.
- **No CUDA arm was measured** because none exists; the op is CPU-only and the
  spec carries that under `## Owed`.

### The fresh review's findings, and what each one cost

The review returned `PASS` on the change: the mutations reproduce and all eight
pre-existing suites are count-identical. Six of its eight findings were prose,
records, a dead build define and a merge. The other two changed something
measured — one a published claim, one a dtype — and both are recorded here,
because a reader of this section would otherwise take the earlier text at face
value.

**The "first suite in this tree" claim was FALSE unscoped.** `## Now` said this
was the first suite here to load a gamma through `ModelRegistry::Load` and run it
through a device op in one case. `tests/vllm/models/test_nemotron_h_paged_forward.cpp`
and `tests/vllm/models/test_kimi_linear_paged.cpp` already do both inside a
`TEST_CASE`. Scoped to `qwen4_exp` the claim holds, and the argument it supports
— eleven single-sided waves of THIS row could not see the contradiction —
survives unchanged. Corrected in `## Now`, in `## Owed` and on #2218 itself.

**The four-gamma attribution was wrong about one CONSUMER.** The `## Owed` entry
said `RunQwen4ExpQsaBlock` normalizes all four QSA gammas through
`vt::RmsNorm(gemma = true)` and then cited three line pairs. The count exposed
it: `idx_k_norm` never reaches `vt::RmsNorm`. It goes to `Qwen4ExpQsaIndex`
(`qwen4_exp_qsa_block.cpp:401-403`) and is consumed by `vt::Qwen4ExpQsaCompress`
(`:181`), which adds the 1 itself. Same polarity, different op, so "three of the
four consumers already add the 1" stands with the consumer named correctly.

**THE FOLD'S DTYPE HAD DRIFTED, AND THE BAND WAS ABSORBING IT.** Before this
wave, the wide-accumulator case handed one identical `float` multiplier to both
arms. After it, the kernel folded `1.0f + w` in f32 while the double reference
folded `1.0 + (double)w`, so the two arms no longer described the same multiplier
and the case's own comment — "the only thing this widens is the reduction" —
stopped being true. Nothing failed, which is the point. Measured on exactly the
data in the case, by forcing the bound to `1e-30` and reading the logged `worst`:

| Reference's fold | Worst absolute deviation, `mixed` vs reference |
|---|---|
| `1.0f + w_hf`, widened AFTER (f32, as landed here) | 1.17323e-06 |
| `1.0 + (double)w_hf` (the drifted form) | 9.8457e-07 |

Both sit far inside the band — the bound is `1e-5` and the `float ss` mutant
reads 6.702e-4 — so no tolerance was ever at risk. What was at risk is the
meaning of the number: **1.173e-06 is the figure this file and the W5b-2 table
record as "ours, double accumulator", and the drifted form no longer reproduced
it.** The f32 fold is also what upstream does —
`output * (1.0 + self.weight.float())` (`modeling_qwen4_exp.py:177`) folds a weak
Python `1.0` into an fp32 tensor and the promotion stays fp32 — so mirroring
upstream and restoring the recorded measurement are the same edit. AGENTS.md
"Inherit vLLM defaults" decides it either way: f32 is the default and the wider
value would have been the annotated exception, unannotated.

## Mutation record — W5d-2 (#2249 item 5)

The interleaved-mRoPE cos|sin table builder, `BuildMropeCosSinHost`. It was
`static` at `src/vllm/model_executor/models/qwen3_5.cpp:9472`, so the tables
Qwen4-Exp's QSA half of the layer loop needs could not be built from another
translation unit and the QSA block would have had to grow a second copy of the
axis selection and the angle math.

**WHICH SHAPE, AND WHY THE SIMPLER ONE.** `RunGdnBlockPaged` (W5b-1) and
`RunMoeBlock` both needed a thin PUBLIC WRAPPER over a private definition,
because their signatures name types qwen3_5.cpp declares privately
(`StepDevInputs`). This one names nothing private — `std::vector<int32_t>`,
`int64_t`, `vllm::HfConfig` — so the extraction is the `static` keyword and a
declaration in `include/vllm/model_executor/models/qwen3_5_mrope.h`. The
definition does not move and there is exactly ONE implementation: qwen3_5.cpp's
own two call sites now resolve through the same public declaration qwen4_exp
will use, which is what `AGENTS.md` `## Shared seams` requires and what a
copied second table builder would have broken.

**BYTE IDENTITY, TWICE.** First textually: `git show
94de63ff5:src/vllm/model_executor/models/qwen3_5.cpp | sed -n '9473,9514p'`
sha256s to `259b1b932cae0611ca6dbde4ad63214e0d1365efe3b708b8ef7d38a7894688f1`,
and so does the body on this branch — the whole diff to that function is the
`static` keyword and two comment lines. Second by VALUE, because the keyword
that changed is exactly the one that decides which definition a caller binds
to: `tests/vllm/models/test_qwen3_5_mrope.cpp` pins 152 f32 BIT PATTERNS across
four cases against what the FILE-STATIC produced at base SHA `94de63ff5`,
captured by compiling its `sed`-extracted text in a standalone harness. The
comparison is bitwise and not an epsilon — this is a pure host computation over
`std::cos`/`std::pow` with no reduction-order freedom, so a tolerance would hide
the one defect an extraction can introduce.

**Counts, before and after, on the same tree.** 26 pre-existing qwen3_5 /
qwen4_exp suites built and run at base and at head, identical exit status and
identical case and assertion counts on every one (`diff` of the two count files
is empty). The new suite adds 4 cases / 157 assertions. The population is every
`vllm_cpp_add_test` target in `tests/CMakeLists.txt` whose name matches
`qwen3_5`, `qwen35` or `qwen4_exp`, less the benchmark
`bench_qwen3_5_vl_tower` and less this wave's own `test_qwen3_5_mrope`.

**WHICH TREE THAT 26 WAS COUNTED ON, because merging `main` moved it.** The
count is base `94de63ff5` against branch head `c1ccbac19`, both of which
predate the merge of `main` in this branch. That merge brings in W5b's
`test_qwen4_exp_forward` ([#2031](https://github.com/mudler/vllm.cpp/issues/2031),
landed on `main` as `a6f933b81`'s neighbour), which makes the same glob match 27
targets on the merged tree. It is NOT a 27th row of this neutrality
measurement and cannot be: it existed at neither end of the before/after pair,
so it has no before. It is `main`'s own gate for `main`'s own wave. The 26 is
therefore a statement about the two trees named here and not about the merged
head, which is the distinction this section previously left for a reader to
make.

**FOUR of the 26 measure NOTHING on this host, and only one of them says so.**
An earlier revision of this section said "23 suites, two of which do not
measure". Both halves were wrong. Re-measured on this CPU-only host at this
head:

| suite | rc | cases | assertions | why it measures nothing |
|---|---|---|---|---|
| `test_qwen35_paged_engine` | 77 | — | — | prints `*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***`; the Qwen3.5-0.8B snapshot at revision `2fc06364` is not cached here. **This is the one that is honest about itself** |
| `test_qwen35_gguf_spec_decode` | 0 | 3 | **0** | `SKIP: set VLLM_MTP_GGUF_MODEL` |
| `test_qwen3_5_vl_e2e` | 0 | 1 | **0** | `SKIP: Qwen3.6-27B checkpoint absent (set VLLM_QWEN36_CKPT)` |
| `test_qwen3_5_vl_video_e2e` | 0 | 1 | **0** | the same skip |

The last three exit 0 and print `[doctest] Status: SUCCESS!`. That is a skip
wearing a pass, and a count-diff over a population containing them is neutral by
construction on those three rows, so they carry no neutrality evidence at all.
They are listed so that a reader does not read 26 green suites as 26
measurements.

**AND THIS BOUNDS THE M3 REACHABILITY EVIDENCE, WHICH IS THE PART THAT MATTERS.**
`test_qwen3_5_vl_e2e` and `test_qwen3_5_vl_video_e2e` are the STRICT token-exact
end-to-end gates on `VLGenerateCoreGdn`, the shared driver core holding the two
production call sites M3 deletes. On a host that has the Qwen3.6-27B checkpoint
they would be the strongest witnesses M3 has. Here they measure nothing, so the
M3 red rests ENTIRELY on `test_qwen3_5_moe_vision` (7 cases / 38 assertions),
whose `qwen3_5_moe_vl_image_forward_uses_MRoPE_positions_not_plain_1d` is the
single case that goes red. One case, one assertion, is the whole reachability
proof on this host. `test_qwen3_5_moe_vision_hw` does not extend it either: it
measures 3 cases / 23 assertions but its own e2e case skips on
`VLLM_MOE_VISION_E2E`. This is a HOST condition and not a defect in the
mutation — it is stated because a reader on a GPU host with the checkpoint gets
strictly more evidence than this run produced, and a reader without it gets
exactly one assertion.

**Upstream.** No divergence found. The interleaved axis masks mirror
`vllm/model_executor/layers/rotary_embedding/mrope.py:60-63` at the parity pin
`5559679229` (`h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 *
mrope_section_h)`, and the `w` twin), `apply_interleaved_rope` (`:190-198`)
states the same layout as a tensor rewrite, and the chunked branch mirrors the
same function's `else` arm (`:66-70`). The per-pair frequency is
`base ** (-2 * pair / rotary_dim)`, which is `RotaryEmbeddingBase`'s inv_freq.

| # | mutation | build rc | target | result |
|---|---|---|---|---|
| M1 | swap the cos and sin stores for the `h` axis (`axis == 1`) inside the extracted function | 0 | `test_qwen3_5_mrope` | **RED, 4 of 4 cases, 42 of 157 assertions.** The first failures are index 1 and index 9 of C1 trading values, which is the swap seen directly |
| M2 | change the position offset by one (`positions3[axis * T + i] + 1`) | 0 | `test_qwen3_5_mrope` | **RED, 4 of 4 cases, 141 of 157 assertions.** The 16 survivors are the pairs whose frequency is small enough that the f32 store absorbs one position |
| M2b | `pair <= 3 * sec[1]` -> `pair < 3 * sec[1]`, the upstream `<=` | 0 | `test_qwen3_5_mrope` | **GREEN — an EQUIVALENT MUTANT, and provably so.** The two forms differ only at `pair == 3 * sec[1]`, and the guard already requires `pair % 3 == 1` while `3 * sec[1]` is divisible by 3, so no input separates them. Upstream's `<=` and a `<` are the same function here. Recorded rather than replaced by a stronger case, because the next reader will reach for this mutation too |
| M2c | shift the same boundary instead: `pair <= 3 * sec[1] + 1` | 0 | `test_qwen3_5_mrope` | **RED, 2 of 4 cases, 8 assertions.** This is the section boundary actually under gate: on C1 (`sec = {4,2,2}`, half 8) pair 7 flips from the `t` axis to the `h` axis |
| M2d | the SAME shift on clause TWO, the `w` axis: `pair % 3 == 2 && pair <= 3 * sec[2]` -> `... + 2`. `+2` and not `+1`, because `3 * sec[2]` is divisible by 3 and the clause already requires `pair % 3 == 2`, so `+1` would be a second equivalent mutant for exactly M2b's reason | 0 | `test_qwen3_5_mrope` | **RED, 1 of 4 cases, 2 of 157 assertions.** Added on review repair, because M2b's green is only honest if the OTHER clause's reachable boundary is shown to red too — otherwise a reader cannot tell an equivalent mutant from an ungated one. Pristine `qwen3_5.cpp` sha256 `0b4517b3246e6e49fd8b0fa3a8ad7adc5c39b2846a4800966733688fb0d8d9fe` before, `c00f7a461b65a3260ea255b30bc03864bd7ad53cfb9379905cc41fe38b10ff8f` under the mutation, and back to `0b4517b3…` on restore; BUILD RC 0 read before the test result on both legs; re-run green 4 of 4 cases / 157 of 157 assertions, and `test_qwen3_5_moe_vision` 7 of 7 / 38 of 38 |
| M3 | REACHABILITY: delete both production call sites in `qwen3_5.cpp` (`VLGenerateCoreGdn`'s prefill build and its decode-continuation build) and pass `nullptr` for the cache | 0, after a `(void)` for `-Werror=unused-parameter` | `test_qwen3_5_moe_vision`, `test_qwen3_5_mrope` | **`test_qwen3_5_moe_vision` RED on exactly `qwen3_5_moe_vl_image_forward_uses_MRoPE_positions_not_plain_1d` (1 of 7 cases, 1 of 38 assertions)** — the VL greedy driver reaches the extracted function and a test enters through the driver. It is NOT a production entry point; see the paragraph below. **`test_qwen3_5_mrope` stays GREEN, and it must:** it is a unit and seam case that calls the function directly, so it measures the function and never that anything reaches it. Stated here rather than left to be inferred |

**WHAT M3 DOES NOT PROVE, MEASURED RATHER THAN ASSUMED.** The chain M3 reds
stops one hop short of a production entry point, and this wave did not create
that and does not close it. `grep -rn 'Qwen3_5MoeVLGenerateGreedy|Qwen3_5VLGenerateGreedy'`
over `src/ include/ examples/ tools/ benchmarks/` returns the four DEFINITIONS
in `src/vllm/model_executor/models/qwen3_5.cpp:9892,9915,9960,9974` and their
six declaration lines in `qwen3_5.h` / `qwen3_5_dense.h` — and NOTHING else.
Every CALLER is in `tests/`. The registered factories for
`Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`
carry no multimodal hook, so `ModelRegistry::Forward` cannot arrive here, and
`include/vllm/entrypoints/openai/chat_mm.h:266-267` already says so in the tree's
own words for the sibling Qwen3-VL driver: the greedy VL drivers run "outside
`ModelRegistry::Forward`". So `BuildMropeCosSinHost` is reached by a public,
gated, non-test caller, and that caller is not yet routed from
`include/vllm.h`, the loader, `ModelRegistry::Forward` or a server path. The
extraction changes nothing about that either way — the function had exactly this
reach before the `static` came off — and it is recorded under `## Owed` rather
than left for a reader to discover, because `AGENTS.md` `## Nothing lands dead`
asks the question at every merge and silence is not an exception.

Every mutation was applied to a pristine `qwen3_5.cpp`, sha256-proven applied,
rebuilt with the BUILD RC read BEFORE any test result, run, then restored from a
byte-identical copy and re-proven at
`0b4517b3246e6e49fd8b0fa3a8ad7adc5c39b2846a4800966733688fb0d8d9fe`, rebuilt and
re-run green. M3's first attempt did NOT build — deleting the call leaves
`pos3_prefill` unused under `-Werror=unused-parameter` — which is the W5b-1
mutation-B trap again and the reason the build rc column is in this table.

## Mutation record — W5d-1 (#2249 item 1)

`vt::RmsNormGroup` / `OpId::kRmsNormGroup`, the ungated per-group RMS norm the
PLE half of the layer loop needs and the one primitive `include/vt/ops.h` named
as missing in its own words.

**THE TREE EVERY NUMBER BELOW WAS MEASURED ON**, because an evidence table that
does not name its tree is not evidence. The RED, the green, the six mutations
and the suite counts were all taken at base `94de63ff5`. The branch was then
rebased forward twice as `main` moved under it, onto `6e805abcf` (`QUANT-EXL3`
W3 — `cpu_exl3_kernels.cpp`, `cuda_exl3.cu`, `test_exl3_gemm.cpp`,
`dense_weight_loaders.h`) and then onto `5f8a70705` (`SPEC-DFLASH2` #2252 —
`qwen3_dflash*`). Neither touches a file this op compiles against.
`test_ops_rms_norm_group` was rebuilt and re-run on that head: **build rc 0,
7 cases / 69 assertions / rc 0**, unchanged. The mutation battery was NOT re-run
after either rebase, and that is stated rather than implied.

**The review repair then MERGED `origin/main` into the branch** rather than
rebasing a third time, and has now had to do it THREE TIMES, because `main` kept
moving while the repair ran. `scripts/agent-preflight.sh` skips both trailer
gates whenever `origin/main` is not an ancestor of HEAD — "this branch is behind
it and the trailer gates did NOT run" — which had quietly voided the review's
own `PREFLIGHT_RC=0`. The commits merged in are `1bc16ca3c`
(`PERF-LAGUNA-GROUPED-GEMV` spec) and `37fbccea8` (`MODEL-TEXT-GLM-MOE-DSA` spec)
first, then `fa9903b86` (`LTX25-ORACLE-ABSOLUTE`, #2210), and finally `3ed2378a3`
(W5d-2, #2249 item 5, via #2264). THE FIRST TWO touch `.agents/`,
`docs/USAGE.md` and two `scripts/` files and NO compiled input: `git diff
--name-only` over that delta returns nothing under `src/`, `include/`, `tests/`
or `third_party/`, and `ninja` answered "no work to do" after it.

**THE THIRD MERGE IS DIFFERENT, WHICH IS WHY THE SENTENCE ABOVE IS SCOPED TO THE
FIRST TWO RATHER THAN LEFT TO COVER ALL THREE.** The delta `fa9903b86..3ed2378a3`
is fourteen commits and it DOES move compiled input, including both files this
wave itself edits: `include/vt/ops.h`, where W5b-6 rewrote the
`vt::Qwen4ExpGatedResidual` contract comment in place (#2218 via `a6f933b81`),
and `tests/CMakeLists.txt`, where W5d-2 registered `test_qwen3_5_mrope`. Git
merged both without a conflict and both merges are purely ADDITIVE against
`main`: `git diff origin/main` over the two files shows this wave's blocks and
nothing removed. The one consequence a conflict-free merge could still have hidden
is an id shift, because `kRmsNormGroup` is appended before `kCount` and so is
every other new op — nothing on `main` appended an enumerator, and the merged
header compiles to `kRmsNormGroup == 140`, `kCount == 141`, so the `id 140` the
refusal prints below is still this op's id. `.agents/specs/qwen4-exp-flash-next.md`
did conflict and was resolved as a SET against the merge base rather than by
taking either side, and the one anchor the merge moved is corrected under
`## Owed`: the `kTENSTORRENT` `RegisterOp(OpId::kRmsNorm, ...)` line is at
`tenstorrent_ops.cpp:5323` on this head, not the `:5216` recorded before
`BACKEND-TENSTORRENT-QWEN35` W5/W6 landed.

**A false alarm is recorded here rather than buried, because it nearly landed a
duplicate.** `origin/main` is a shared ref in a shared checkout, and another
session fetched it mid-repair. Comparing the merged index against the ref AFTER
that fetch read as though the first merge had silently dropped its tail row
(#2220), and a commit was written to "restore" it. The merge had dropped nothing:
the row arrived with `fa9903b86`, which had not been merged yet. Appending it by
hand would have produced a SECOND copy of that row on `main` — the duplicate
`check-agent-record.py` refuses and `check-issue-index-append-only.py` will not
permit anyone to remove. The commit was dropped before it left the worktree.
`check-issue-index-append-only.py` returned rc 0 on BOTH the pre- and
post-"restore" heads, so the gate would not have caught it either way: the
control that worked was diffing the row-number list against the ref and asking
WHICH commit authored the row, not the checker.

At the THIRD merged head, from a build directory configured from scratch, build
rc 0 read before any test result, **the same FIVE of the seven suites below were
re-run and all five still match byte-for-byte**: `test_ops_rms_norm_group`
7 / 69, `test_ops_mamba2_gated_norm` 9 / 2107, `test_ops_glue` 13 / 115,
`test_qwen4_exp_hc` 15 / 246 and `test_qwen4_exp_hc_device` 9 / 87. Because that
merge brings compiled input with it, `main`'s OWN two new gates were built and
run here as well, and both reproduce the counts `main`'s own sections record:
`test_qwen4_exp_forward` 1 / 421 (W5b-6, #2218) and `test_qwen3_5_mrope` 4 / 157
(W5d-2, #2249 item 5). `test_qwen4_exp_scaffold` was re-run as well, at
12 / 296 / rc 0, because this merge EDITS the production refusal string that
suite's `SUBCASE("the forward")` pins — see `## Now` for why the string had to
change and which five substrings the suite holds. `test_qwen4_exp_ple` and
`test_qwen4_exp_ple_device` were NOT re-run at any merged head, and their rows
below still name `94de63ff5`.

**M4 and M5 were re-run at the SECOND merged head**, because the `## Owed`
sentence that repair corrects is a claim about exactly those two: M4 build **rc 1** with
`error: 'RmsNormGroupKernel' defined but not used [-Werror=unused-function]` and
NO suite run; M5 build **rc 0**, suite **rc 1**, **6 of 7 cases**, six throws of
`vt: no kernel for op RmsNormGroup (id 140) on device cpu (type 0)` raised at
`src/vt/op_provider.cpp:589`. `src/vt/cpu/cpu_ops.cpp` was restored
byte-for-byte after each, sha256 `e4a924b4…04b5` both times, rebuilt at rc 0 and
re-run green at 7 / 69. **They were NOT re-run at the third merged head**, so the
battery has not been re-measured since `main` began moving compiled input under
this branch; the throw site `src/vt/op_provider.cpp:589` and the printed
`id 140` were both re-checked there, the battery itself was not. Every other
number in this section still names `94de63ff5` and is not restated as if it were
measured here.

Method as in the sections above:
one textual change applied to a pristine tree, proved applied by a **sha256 that
moved**, the file `touch`ed so ninja cannot skip the rebuild, the **BUILD RETURN
CODE READ BEFORE ANY TEST RESULT**, then restored from a byte-identical copy and
`sha256sum`-verified against the pre-mutation digest. Suite
`tests/vt/test_ops_rms_norm_group.cpp`, **7 cases / 69 assertions / rc 0** green
at the head this table was measured on.

**WHY IT IS A NEW OpId AND NOT A FIELD ON `RmsNormArgs`.** `kRmsNorm` is
registered on more than one backend. A `group_size` added to its shared args
struct is IGNORED by every kernel not taught to read it, so a device whose
kernel was not updated would answer a grouped request with a whole-row norm —
no crash, no refusal, a plausible tensor. A separate OpId cannot fail that way:
an unregistered device refuses BY NAME, which M5 below measures. `kRmsNormGatedGroup`
is the in-tree precedent for exactly this split.

### The RED, before the change

The test file written first, against the tree at `94de63ff5`, compiling the test
translation unit alone (`ninja tests/CMakeFiles/test_ops_rms_norm_group.dir/vt/test_ops_rms_norm_group.cpp.o`):

```
BUILD_RC=1
test_ops_rms_norm_group.cpp:75:11: error: 'RmsNormGroupArgs' has not been declared in 'vt'
test_ops_rms_norm_group.cpp:120:7: error: 'RmsNormGroup' is not a member of 'vt'; did you mean 'RmsNormGated'?
```

The compiler's own suggestion is the gap in one line: the nearest thing this tree
had is the op that cannot express it. That red is a BUILD refusal and is read as
one — it says the op is absent, not that any arithmetic is wrong. The red for the
intended REASON is M1, which builds clean and fails on values.

### The gate, and why each half of it discriminates

The correctness assertions run against `tests/vllm/models/qwen4_exp_hc_goldens.inc`,
whose `k{A,B,C,D}_normed` arrays are `normed = mod.hc_norm(hyper)` — the pinned
oracle's OWN `Qwen4ExpTextRMSNorm(hc*hidden, group_size=hidden, eps)` output over
its own RAW gamma, dumped by `scripts/gen-qwen4-exp-hc-goldens.py` from
transformers **v5.16.0** (`modeling_qwen4_exp.py:158-181`, sha256
`77fec77d…c459`). Nothing in the correctness path is transcribed. A local
double-precision reference exists in the file, but ONLY to measure the
separations below; the op is never asserted against it.

| Defect | Separation from the oracle | kTol |
|---|---|---|
| reduce over the ROW, not the group | A 1.232, B 0.984, C 0.404, D 0.913 | 1e-5 |
| drop the `+ 1` on the gamma | A 2.279, B 2.181, C 2.053, D 1.986 | 1e-5 |
| drop eps | A 4.1e-6, B 1.67e-5, C 5.1e-7, **D 2.56e-2** | 1e-5 |

**The eps row is the reason case D exists and the reason an eps probe run at
A's scale is a mute switch.** At `hyper_scale = 1.7` the mean square is O(1) and
an eps of 1e-6 moves the answer by less than the tolerance; at D's
`hyper_scale = 0.01` it is 1% of the mean square. The file asserts BOTH
directions — `sep > 1e2 * kTol` at D and `sep < kTol` at A — so the fixture's
blind spot is recorded executably rather than left for the next reader to
rediscover.

### The battery

| # | Mutation | Build | Result |
|---|---|---|---|
| M1 | `RmsNormGroupKernel`: `group_size = h`, i.e. reduce over the whole row | rc 0 | **RED**, `7 cases / 5 failed`, `69 assertions / 26 failed`. Every value case moves: the oracle case at all four goldens, the four-orders-apart case, the fold case, the eps case and the bf16 rounding case. The two survivors are the two that call no op — the fixture-separation case and the refusal case |
| M2 | drop `if (args.gemma) wj += 1.0f`, the #2218 defect | rc 0 | **RED**, `4 cases failed`, `22 assertions failed`. This is the polarity the whole architecture now shares: every gamma is stored RAW and every consumer adds the 1, `ssm_norm` excepted |
| M3 | drop `+ args.eps` from inside the rsqrt | rc 0 | **RED**, `4 cases failed`, **`5 assertions failed`** — and the small count is the finding, not a weakness. Only goldens B and D move; A and C are BELOW the tolerance, exactly as the table above predicts. A probe placed only at A would have reported this mutation as survived |
| M4 | delete `RegisterOp(OpId::kRmsNormGroup, DeviceType::kCPU, ...)` | **rc 1** | **A BUILD REFUSAL, not a test verdict, and read as such:** `error: 'RmsNormGroupKernel' defined but not used [-Werror=unused-function]`. That registration is the kernel's ONLY reference in the tree, so the compiler proves the link a test result would only have suggested. No suite ran under this mutation |
| M5 | register the same kernel on `DeviceType::kCUDA` instead — the runnable form of M4 | rc 0 | **RED**, `6 of 7 cases` threw `vt: no kernel for op RmsNormGroup (id 140) on device cpu (type 0)`. This is the load-bearing reachability proof at the layer that exists: the suite reaches the kernel THROUGH `GetOp`, not by calling it directly, and the `op_provider.cpp` name entry is live too, because the refusal prints the op by name |
| M6 | delete the `args.group_size >= 1` refusal in the dispatcher | rc 0 | **RED**, rc 136 — `SIGFPE`, an integer divide by zero at `h / group_size`. The refusal is therefore load-bearing rather than decorative, and the default `group_size = 0` is genuinely unusable rather than quietly meaning "the whole row" |

M1-M5 target `src/vt/cpu/cpu_ops.cpp`, M6 `src/vt/ops.cpp`. M6 was re-measured on
the FINAL head after its refusal message was corrected; M1-M5 were measured on a
head that differs from the final one only in that message string, in a file they
do not touch.

### Counts on this head

| Suite | Result |
|---|---|
| `test_ops_rms_norm_group` | **7 / 69 / rc 0** (new; re-run identical on the rebased head) |
| `test_ops_mamba2_gated_norm` | 9 / 2107 / rc 0 |
| `test_ops_glue` | 13 / 115 / rc 0 |
| `test_qwen4_exp_hc` | 15 / 246 / rc 0 |
| `test_qwen4_exp_hc_device` | 9 / 87 / rc 0 |
| `test_qwen4_exp_ple` | 9 / 395 / rc 0 |
| `test_qwen4_exp_ple_device` | 10 / 538 / rc 0 |
| `test_qwen4_exp_qsa` | 14 / 7263 / rc 0 |
| `test_qwen4_exp_qsa_device` | 12 / 4697 / rc 0 |
| `test_qwen4_exp_qsa_block` | 8 / 2831 / rc 0 |
| `test_qwen4_exp_kv_cache` | 4 / 399 / rc 0 |
| `test_qwen4_exp_scaffold` | 12 / 296 / rc 0 |
| `test_qwen4_exp_gguf_weights` | 11 / 2975 / rc 0 |
| `test_qwen4_exp_gguf_load_plan` | 10 / 7462 / rc 0 |

**The BEFORE column is stated rather than re-measured, and the reason is
checkable.** `git diff --numstat` over `include/` and `src/` is `118/0`, `52/0`,
`2/0`, `31/0` — **zero deletions, zero modified lines**: a new enumerator before
`kCount`, a new args struct, a new function-pointer alias, a new declaration, a
new kernel with its registration, and a new name case. No existing behaviour is
reachable from any of it. Four of the rows above are additionally cross-checked
against numbers this spec already recorded before this wave — `test_qwen4_exp_qsa`
14 / 7263, `test_qwen4_exp_qsa_block` 8 / 2831, `test_qwen4_exp_kv_cache` 4 / 399
and `test_qwen4_exp_scaffold` 12 / 296 — and all four match exactly.

**What the battery did NOT reach**, because a battery's silence is not a result:
no CUDA arm exists to mutate; no production entry point calls the op, so no
mutation here can measure a reach that does not exist (`## Owed`); and the group
widths exercised are 4, 5 and 6, not the 2560 the released config uses, so the
f32 sum-of-squares accumulator is gated at toy width only.

## Mutation record — W5d-3 (#2249 item 2)

The wave that gave the QSA consumer a PAGED read path. Measured on an
`origin/main` base of `fa9903b860`, CPU only, Release, `-j 2`. The build return
code was read BEFORE any test result on every row, because a failed build reads
as a passing test.

**The instrument.** The paged cases fill the WHOLE flash cache with bf16 NaN
before the block runs, so every row a correct read never addresses — an unnamed
physical page, or the unused tail of the last named one — is not a number. That
is the same discriminator the W5b-4 gather-vs-mask case uses, doing a second job:
there `0.0f * NaN` convicts a MASK, here it convicts a wrong ADDRESS. The block
table is `{5, 3, 7}` against a logical `{0, 1, 2}`, sharing no fixed point, so the
three pages an identity-reading body touches are exactly three that are never
written. **An identity block table would make the whole case prove nothing**, and
that is why the permutation is stated here rather than left to the fixture.

**And one finding about the harness itself, measured rather than feared.**
`MaxRelDiff` folds with `std::max`, and `std::max(x, NaN)` returns `x`. So the
first RED capture below came back ALL NaN and the relative bound still printed
`0` and PASSED. A tolerance cannot see a NaN. The two paged cases therefore assert
FINITENESS FIRST and the oracle bound second, and the test says why.

| # | Mutation | Site | Build rc | Result |
|---|---|---|---|---|
| RED | the pre-W5d-3 body: `RowBase` always returns the CONTIGUOUS address `(p * HKV + kvh) * DH`, i.e. the paged arm reads slots linearly | `cpu_qwen4_exp_qsa.cpp` | 0 | **RED, 2 of 11 cases, 1537 assertions.** `CHECK(std::isfinite(v))` x1472 (every output NaN), `CHECK(differing == 0)` at `paged vs contiguous differing bf16 words 1472 of 1472`, and 64 more on the decode case. This is the capture of the gap #2249 item 2 names |
| M1 | OFF-BY-ONE in the page-table read: `pages[(p + 1) / page_size]` | `cpu_qwen4_exp_qsa.cpp` | 0 | **RED, 2 of 11 cases, 195 assertions.** `rel 0.309431 < 0.03` against the oracle, `differing 703 of 1472`, decode `rel 0.667465`. Note it is NOT all-NaN: an off-by-one lands on a WRITTEN page most of the time, which is precisely why the value comparison and the bit-exact one both have to be there |
| M2 | the partial final page read at FULL page length: the ragged tail runs to `ceil(kv_len / page) * page` instead of `kv_len` | `cpu_qwen4_exp_qsa.cpp` | 0 | **RED, 2 of 11 cases, 515 assertions.** 448 NaN outputs (row 7 of the last page is the one row the 23-token sequence never writes), `rel 1.24777`, `differing 1339 of 1472`, and the `keys_visited` equality with the contiguous arm |
| M4 | the PAGED STORE deleted (`dense_attn::WriteKvCache` never called) | `qwen4_exp_qsa_block.cpp` | 0 | **RED, 2 of 11 cases, 1537 assertions**, all NaN. The store site is gated, not merely present |

Every mutation was sha256-proved applied, and the tree was restored byte-for-byte
after each (`cpu_qwen4_exp_qsa.cpp` back to
`d95eea49e1800a25fb0b920a43c92936973c5e51e576651fa74400064a2497cc`,
`qwen4_exp_qsa_block.cpp` to
`feb0eccd41d39a1588a9ffd32db9bfec50ac01dc40e44e494ad426d7ca3c43b7`).

**M3, the reachability mutation, HAS NO SITE, and that is the finding rather than
an omission.** `.agents/reachability.md` asks for the production call site to be
deleted in a scratch copy. `grep -rn 'RunQwen4ExpQsaBlock\|Qwen4ExpQsaPagedCaches'
src include examples` returns only the block's own header and translation unit:
there is no production caller to delete, because
`ForwardQwen4ExpForConditionalGeneration` still refuses by name and the layer loop
is unwritten. So what these mutations measure is a CAPABILITY of the block, not
that anything reaches it. `## Owed` records the wave as UNREACHED with the owning
row and the issues.

**What the battery did NOT reach.** The device (CUDA) arm of the address mode does
not exist, so nothing here says a GPU resolves a page the same way. Nothing decodes
a real checkpoint through the paged arm. And no mutation here can see a wrong
INDEXER side-cache address, because that cache is still contiguous — #2249 item 3,
owed as W5c-2.

## Mutation record — W5c-2 (#2249 item 3)

The group-2 block-table gather. `GPUModelRunner::gather_block_table` had three
call sites and reached exactly two group ids, so a model publishing a THIRD
group — `qwen4_exp`'s QSA indexer side cache, an `MLAAttentionSpec` at
`compress_ratio` 4 — had that cache allocated and no map from a logical position
into its pages.

**WHAT WAS MIRRORED, AND WHY IT IS A LOOP AND NOT A THIRD NAMED ID.** Upstream
has no "the two special groups" shape at all. Its per-group metadata build runs
over `enumerate(kv_cache_groups)` and hands each group its own table —
`cm.block_table_tensor = _get_block_table(kv_cache_gid)`,
`vllm/v1/worker/gpu_model_runner.py:2551-2567` at the parity pin
`5559679229` — where `_get_block_table` (`:2318-2334`) is
`self.input_batch.block_table[kv_cache_gid].get_device_tensor(...)`, which is
byte-for-byte what this tree's `gather_block_table` does. Group 0 is gathered
once BEFORE the loop (`block_table_gid_0 = _get_block_table(0)`, `:2337`) and
carried into every iteration on `cm_base`; the loop body itself is guarded by
`if kv_cache_gid > 0:` (`:2565`), so upstream does NOT re-gather group 0 inside
it and this tree does. That is an OPTIMISATION on upstream's side, not a second
convention for "which groups are special": the value is the same table either
way. We pay one extra copy of a table the step already built rather than
special-casing an index, and the extra copy is disclosed here and in the commit
body. `GPUModelRunner::gather_group_block_tables` is that loop.

**WHERE THE TABLES GO, AND WHY NOT ON `CommonAttentionMetadata`.** Upstream fans
its per-group metadata out BY LAYER NAME (`:2551-2552`, "make layers in the same
group share the same metadata"), and this tree's mirror of that key is
`MultiKvCacheIndex` (KV-DSV4-MULTICACHE W3, #2068) — the channel that already
carries which group each published cache came from. The two new vectors are
indexed by GROUP ID rather than parallel to `attn_kv`, because a block table
belongs to a group and every layer in it shares one. `CommonAttentionMetadata`
carries exactly one table, the target group's, and widening it would put a
multi-cache field on every uniform step.

**WHAT IS STILL NOT REACHED, STATED BEFORE THE BATTERY.** The gather runs on the
production `execute_model` path and its count is READ by the multi-cache refusal
in `ModelRegistry::Forward`. Nothing CONSUMES the tables: no forward reads them,
because `ForwardQwen4ExpForConditionalGeneration` refuses by name and
`ModelRegistry::Forward` refuses every multi-cache topology. That is carried
under `## Owed` and named in the commit and pull-request bodies.

### The RED, before the change

The carrier landed first (the two `MultiKvCacheIndex` fields, the two accessors
and the refusal's new clause) with NO gather wired, so the red is behavioural
rather than a compile error — the accessor exists and answers `nullptr`:

```
tests/vllm/v1/worker/test_runner.cpp:2498: ERROR:
  CHECK( msg.find("block tables gathered for 3 of 3 published group(s)")
         != std::string::npos ) is NOT correct!
  values: CHECK( 18446744073709551615 != 18446744073709551615 )

tests/vllm/v1/worker/test_runner.cpp:2508: FATAL ERROR:
  REQUIRE( bt != nullptr ) is NOT correct!
  values: REQUIRE( nullptr != nullptr )
  logged: g := 0
```

### Why the fixture discriminates

`MakeQwen4ExpShapedKvConfig()` is this row's own miniature: `FullAttentionSpec` +
one uniform `MambaSpec` + the indexer `MLAAttentionSpec`. The case drives
`GPUModelRunner::execute_model` — the production entry point — with a 20-token
prompt over a 16-token block, so the sequence spans TWO blocks with a PARTIAL
final one, and gives each of the three groups a DISTINCT two-block list with no
fixed point: `{6, 2}`, `{4, 7}`, `{5, 3}`. An identity table would have proven
nothing, because a body that ignores the table, returns the logical indices, or
reads another group's table agrees with `{0, 1}`. The expectation is the literal
list handed to `add_row`, never a value read back out of the runner.

### The battery

Each mutation is a single edit to `src/vllm/v1/worker/gpu/runner.cpp`, applied to
a file whose pre-mutation sha256 is
`b1fcc71a36ee1ac02b87cbc8786958f301177661f3ce07287911d5785c3ae889`, built to a
recorded rc BEFORE any test was run, and restored to that same sha256.

| ID | Mutation | Build rc | Result |
|---|---|---|---|
| M1 | gather every group with the WRONG id — `gather_block_table(gdn_group_id_, …)` instead of `(g, …)` | 0 | **RED**, 6 of 44: groups 0 and 2 both read `{4, 7, 0, …}` where `{6, 2, …}` and `{5, 3, …}` are owed, and both inequality assertions fire |
| M2 | OFF-BY-ONE in the group index — `g == 0 ? 0 : g - 1` | 0 | **RED**, 4 of 44: group 1 reads group 0's `{6, 2}` and group 2 reads group 1's `{4, 7}` |
| M3 | REACHABILITY — the production call site `if (multi_cache_topology_) gather_group_block_tables(num_reqs);` in `execute_model` DELETED | 0 | **RED**, 2 of 1001 over the whole suite: the refusal reports no gather and every group answers `nullptr` |

M3 is NOT vacuous: there IS a production call site and deleting it reds the gate.
What it proves is bounded, and the bound is worth writing down — it proves the
RUNNER reaches the gather, not that any forward reaches the tables. No forward
does; see the paragraph above and `## Owed`.

### Counts, before and after, on the same tree

`tests/test_runner`, one binary, the two new cases excluded by name to get the
"before" figure rather than rebuilding a second tree:

| | test cases | assertions |
|---|---|---|
| before (`-tce` both new cases) | 32 | 990 |
| after | 34 | 1038 |

`tests/test_qwen4_exp_scaffold` is unchanged at 12 cases / 296 assertions, which
is the count its own `## Owed` paragraph in `qwen4_exp_registry.cpp` records.

### The refusal string was falsified by this change, and repaired in it — AND NEITHER SIDE OF THE MERGE WAS TRUE, TWICE

`ForwardQwen4ExpForConditionalGeneration`'s message enumerated "(2) reach for the
indexer side cache, whose group-2 block table `GPUModelRunner::gather_block_table`
never gathers (W5c-2)". That clause describes this commit's PARENT.

**Two waves landed on `main` between the battery above and the merge, and each
rewrote the same literal.** W5d-4 (`3f9177f7f`) removed the MoE clause; W5d-3
(`787373626`, #2276) removed the paged-QSA-consumer clause and left the group-2
clause standing. So at the final merge `main`'s literal enumerated ONE item — the
group-2 block table this change closes — and this branch's literal enumerated ONE
item — the paged QSA consumer W5d-3 closed. Each side was exactly one item too
long, and taking EITHER side verbatim would have put a false statement on `main`.
The literal is therefore resolved BY HAND, and the resolved count is the SET
DIFFERENCE of the five-item survey against every landed wave: **ZERO**. Read off
the tree rather than off either side's prose — item 1 is `vt::RmsNormGroup`, item
2 is `RunQwen4ExpQsaBlockPaged`, item 3 is
`GPUModelRunner::gather_group_block_tables`, item 4 is `qwen4_exp_moe.{h,cpp}`,
item 5 is `BuildMropeCosSinHost` declared in `qwen3_5_mrope.h`; all five resolve.
The enumeration is not DELETED, because a survey that falls silent reads as an
unfinished one: the message says the prerequisites are done and names what is
left, which is the LAYER LOOP itself (#2031) plus the QSA indexer side cache's
paged STORE, and it still refuses because #2031 is unwritten.

The emitted bytes were read back out of the running hook ON THE MERGED HEAD (a
temporary `MESSAGE` in the scaffold subcase, removed and the file restored to its
HEAD sha256
`32ee46d64b3045cbf852c387a2cb106bc46d391ed9fd19b4b6cbc013afac9b07`), not grepped
out of the source — `test_qwen4_exp_scaffold.cpp:767` pins substrings of that
string and therefore pins its PRESENCE, never its truth, and all five of its
substrings survive every wrong variant of this literal. What the hook emitted:

```
vt: Qwen4ExpForConditionalGeneration: the forward is not ported yet. The ops
and block seams ARE on main (W2/W3/W4/W6a/W5a/W5b-1..6, W5c-1, W5c-2, W5d-1,
W5d-2, W5d-3, W5d-4), and with W5c-2 ZERO of the five prerequisites #2249
surveyed remain: GPUModelRunner now gathers every published group's block table,
so the map into the QSA indexer side cache's pages reaches the forward. What is
missing is the LAYER LOOP itself, Qwen4ExpTextModel::Forward, owned by #2031,
with that side cache's PAGED STORE still owed beside it. ModelRegistry::Forward
additionally refuses any multi-cache topology by name, and this model publishes
one. See .agents/specs/qwen4-exp-flash-next.md and issues #2031 and #1978. at
.../src/vllm/model_executor/models/qwen4_exp_registry.cpp:266
```

The `vt: ` prefix and the trailing ` at <path>:266` are what `VT_CHECK` adds; the
line breaks above are presentational and the emitted string is one line, 794
characters as captured. The `:266` is the refusal's live line on this merged head
and is the anchor the `## Owed` bullet above cites. The suite printed
`test cases: 12 | 12 passed` and `assertions: 296 | 296 passed` on the same run,
so reading the bytes cost the gate nothing.

## Mutation record — W5e-1 (#2336)

The PLE GATE as `vt::Qwen4ExpPleGate`. Base SHA `bd90b92b0`. Nine mutations, six
recorded as verdicts and three withdrawn as instrument failures; one
reachability mutation recorded as VACUOUS. Every applied mutation is
sha256-proven applied, every BUILD rc is read BEFORE any test result, and the
tree is restored byte-for-byte from a pristine copy with the sha256 printed.

**THE WHOLE BATTERY WAS RE-RUN ON THE REVIEW-REPAIR TREE, and the table below
carries that tree's numbers.** The repair adds one case (`a NaN score
PROPAGATES`, F2 below) and the suite moves from 8 / 168 to 9 / 176, so every
denominator moved with it and a table left at the old ones would name a tree it
was no longer measured on. Where a row's number changed, the pre-repair value
the fresh review reproduced is kept beside it in parentheses; where it did not,
there is nothing to keep. Two rows are NEW and both come from the review: M-NaN,
which the reviewer ran, and M-NANGUARD, which pins the guard F2 adds.

### The RED, before the change

The test was written first and built against `bd90b92b0`'s product files, with
the op's header, dispatcher, kernel and name entry reverted to HEAD. The build
refused, naming the op the test enters through:

```
tests/vllm/models/test_qwen4_exp_ple_gate.cpp:133:7: error: ‘Qwen4ExpPleGate’ is
    not a member of ‘vt’; did you mean ‘Qwen4ExpPleConv’?
  133 |   vt::Qwen4ExpPleGate(q, t_o, t_s, t_v, args);
```

with 12 further errors from the same absence, at
`tests/CMakeFiles/test_qwen4_exp_ple_gate.dir/.../test_qwen4_exp_ple_gate.cpp.o`.
That is the intended reason: this row's `vt::` surface had no expression of
`modeling_qwen4_exp.py:1181-1182`. `git grep -n
'clamp_min\|signed_sqrt\|SignedSqrt\|copysign' src/vt include/vt` returns zero
lines at that SHA, which is the same fact stated the other way.

### Why the fixture discriminates

The gate is section J of `qwen4_exp_ple_goldens.inc`, produced by `exec`ing
`:1180` alone, then `:1181-1182`, then the `:1184` flatten, VERBATIM by line
range out of transformers v5.16.0 (sha256 `77fec77d…c459`, fetched and hashed in
this flow). The generator reproduces the 483 committed lines BYTE-IDENTICALLY
before appending 91, so section J is an addition and not a regeneration.

`clamp_min(1e-6)` is applied BEFORE the square root, so its whole effect is a
1e-3 floor on |gate| and its whole dynamic range on the output is
`sigmoid(1e-3) - sigmoid(0) = 2.5e-4` per unit of `value`. A fixture on which it
never bound would pass with the clamp deleted — the blind spot #2272 records.
Three of the twelve `(t, j)` pairs are therefore built under the floor and nine
above it, the population is recorded as `kGateClampBinds` READ OFF upstream's own
`:1180` output rather than asserted in prose, and the generator additionally runs
the same three upstream lines with the floor taken to zero and emits the measured
separation, `kGateClampSeparation = 1.5595e-3` — 156x the 1e-5 bound. The test
case `the fixture actually probes the clamp, in both directions` re-checks all of
it, so a future regeneration that stopped probing could not pass in silence.

`(0, 0)` is the ORIGIN and is its own case: the key row is zeroed, so the dot is
EXACTLY 0, `torch.sign(0) == 0` cancels the floor, and the gate is 0 rather than
1e-3. Upstream is genuinely discontinuous there and a fully masked row reaches
it. `the ORIGIN maps to zero, and its neighbours do NOT map to the origin` pins
it in BOTH directions, because each half alone passes a wrong port: `sign(0) ==
+1` moves the origin onto the floor, and an `if (|g| < eps) return 0` shortcut
moves every clamped score onto the origin.

Every comparison routes through `tests/support/max_abs_diff.h`, so a non-finite
operand fails rather than reducing to a perfect 0.0 (#449, #2272). **THAT IS NOW
MEASURED IN THIS SUITE AND NOT INHERITED FROM THE HELPER'S OWN TESTS.** M-NaN in
the battery below poisons the kernel's weight with the operands kept live, and
`max_abs_diff.h:100` fires its `NON-FINITE operand at index 0 … see issue #449`
five times over seven red cases. The previous revision of this paragraph
asserted the property; the row measures it.

**AND THE OP ITSELF NO LONGER SWALLOWS A NaN.** `torch.sign(NaN) == 0` while
`NaN * 0.0 == NaN`, so upstream's `:1181` returns NaN for a NaN score —
confirmed by running the pinned expression under torch on
`[nan, inf, -inf, -0.0, 0.0]`, which returns `[nan, inf, -inf, 0.0, 0.0]`. Our
`SignedSqrt` compared NaN against the clamp (false), tried both sign branches
(false) and fell through to `0.0`, so a NaN score became exactly `0.5 * value`:
#2272's polarity, a poison value rendered as a plausible number, and one
`max_abs_diff.h` cannot catch because by then there is no non-finite operand
left. The fresh review of W5e-1 found it by lifting the function verbatim and
executing it. AGENTS.md §"Mirror vLLM" decides the repair rather than a judgement
about whether the contract admits a NaN: the guard is one line, `if
(std::isnan(g)) return g;`, `a NaN score PROPAGATES` pins it in both directions,
and M-NANGUARD reds that case and only that case when the line is deleted. The
two infinities and the two signed zeros already matched and get no guard.

### The battery

| # | mutation | build rc | result |
|---|---|---|---|
| M1 | the `clamp_min` deleted (`floored = magnitude`) | **1** | **WITHDRAWN — NOT A VERDICT.** `error: unused parameter ‘clamp_min’ [-Werror=unused-parameter]`. No suite ran, so nothing was measured; a build failure that reads as a passing test is the trap `.agents/verification.md` and #2272 both name. Re-run as M1b |
| M1b | the clamp NEVER binds (`magnitude < 0.0 ? clamp_min : magnitude`), which keeps the operand live and removes only the behaviour | 0 | **RED, 5 of 9 cases, 12 of 176 assertions** (5 of 8, 12 of 168 pre-repair). Oracle margin `0.00155973 < 1e-05` — the value `kGateClampSeparation` predicted to FOUR significant figures and not five — `1.5595e-3` from the generator against `0.00155973` here is 0.015% apart, so the agreement is `1559` and the fifth digit differs — measured independently by the generator. The origin case stays correct (`sign(0) == 0` needs no clamp) and its two 1e-12 neighbours collapse onto it, which is exactly the discontinuity the fixture exists to hold |
| M2 | the `sign` multiply deleted (`return root`) | 0 | **RED, 6 of 9 cases, 15 of 176 assertions** (5 of 8, 12 of 168 pre-repair; the NaN case's ORIGIN half is the sixth, because dropping the sign moves that row to `0.50025` too). Oracle margin `0.324619 < 1e-05`; the origin reads `0.50025` where `0.5` is required, and the two "must be distinguishable" checks read `0 > 1e-05` |
| M3 | `sigmoid` applied to the PRE-sqrt gate | **1** | **WITHDRAWN — NOT A VERDICT.** `error: unused variable ‘clamp_min’` and `error: ‘SignedSqrt’ defined but not used`. Re-run as M3b |
| M3b | the same, with `SignedSqrt` still called and its result discarded | 0 | **RED, 5 of 9 cases, 12 of 176 assertions** (5 of 8, 12 of 168 pre-repair). Oracle margin `0.0580857 < 1e-05` |
| M5 | the kernel registered on `DeviceType::kCUDA` instead of `kCPU` | 0 | **RED BY REFUSAL, 7 of 9 cases, 41 assertions reached** (6 of 8 pre-repair; the assertion count does not move, because every case that enters the op now throws before its first `CHECK`). `vt: no kernel for op Qwen4ExpPleGate (id 141) on device cpu (type 0), and the portable CPU reference tier is the SOURCE of that kernel, not a fallback for it`. The dispatcher path and the `op_provider.cpp` name entry are both live rather than vestigial. Same reading W5d-1's M5 carries |
| M-NaN-a | the weight poisoned outright (`weight = std::nan("")`) | **1** | **WITHDRAWN — NOT A VERDICT.** `error: unused variable ‘g’`, `error: unused variable ‘clamp_min’`, `error: ‘Sigmoid’ defined but not used`. The THIRD time `-Werror` refused a naive deletion in this battery, for the same reason M1 and M3 did: this kernel has no dead operands. Re-run as M-NaN |
| M-NaN | the weight poisoned with the operands KEPT LIVE (`Sigmoid(SignedSqrt(g, clamp_min)) * std::nan("")`) | 0 | **RED, 7 of 9 cases, 124 of 181 assertions** (the reviewer measured 6 of 8 and 120 of 173 pre-repair). `tests/support/max_abs_diff.h:100: ERROR: max\|diff\|: NON-FINITE operand at index 0 (got = nan, want = 0.28212). A NaN here used to reduce to 0.0 and PASS — see issue #449` fires **FIVE** times (the two `\|` are escaped for this table; the emitted bytes carry bare pipes). THIS IS THE ROW THAT CLOSES THE #2272 GAP: the finiteness guard is measurably ARMED in THIS suite rather than inherited from `max_abs_diff.h`'s own tests, so "a NaN cannot reduce to a passing 0.0 here" is a measurement and no longer a caveat |
| M-NANGUARD | the `isnan` guard in `SignedSqrt` deleted (F2's repair reverted) | 0 | **RED, 1 of 9 cases, 4 of 176 assertions** — `a NaN score PROPAGATES` alone, and all four of its NaN columns, `CHECK( std::isnan(...) ) is NOT correct! values: CHECK( false )` at `:297`. NOTHING ELSE MOVES, which is the point: the guard is reachable ONLY from a NaN score, so it cannot have shifted a finite answer. Its ORIGIN half stays green, so the guard is not leaking finite scores out of the op either |
| M4 | **the reachability mutation: VACUOUS, and recorded as vacuous rather than as a pass** | n/a | There is no production call site to delete. A tree-wide grep over `src/ include/ examples/ tools/ benchmarks/` finds only the op's own declaration, dispatcher, kernel, registration and name entry, plus two prose mentions in the refusal that names it unreached. `.agents/reachability.md` step 5 distinguishes this from a green gate and `## Owed` carries it |

M1, M3 and M-NaN-a are kept in the table rather than replaced silently, because
the three build refusals ARE the finding: the naive deletion or short-circuit of
any of these behaviours leaves `clamp_min`, `g` or `Sigmoid` unreferenced,
`-Werror` stops the build, and a driver that read the suite's absence as success
would have recorded three false SURVIVEs.

Restore proof: the kernel's sha256 is
`78cdbee1cbec809e616fe1bd113ae8511aa7b0708bca528c1e7d6a2d747f2242` before every
mutation and after every restore, printed by the driver on each iteration, and
the suite is rebuilt and re-run green at the end of the battery (9/176, rc 0).
That hash is the REPAIRED kernel, i.e. the one carrying F2's `isnan` guard; the
pre-repair battery ran against
`e5b00b177d859592cc1f7f940ae2ff6c3ecc7627af274ccb32e7985812aec07f` and is the
tree the fresh review reproduced.
`touch` after each restore, because `cp -a` preserves the mtime and ninja then
SKIPS the rebuild — that trap cost one false link failure in this flow before it
was caught.

### Counts, before and after, on the same tree

| suite | at `bd90b92b0` | at this head |
|---|---|---|
| `test_qwen4_exp_ple_gate` | did not exist | 9 / 176 / rc 0 |
| `test_qwen4_exp_ple` | 9 / 395 / rc 0 | 9 / 395 / rc 0 |
| `test_qwen4_exp_ple_device` | 10 / 538 / rc 0 | 10 / 538 / rc 0 |
| `test_qwen4_exp_scaffold` | 12 / 296 / rc 0 | 12 / 296 / rc 0 |

The scaffold suite is unchanged although the refusal string it drives was
rewritten, which is the point `## Owed` already makes about it: it pins five
substrings as PRESENT and cannot pin any of them TRUE. The repair was therefore
verified by READING THE EMITTED BYTES out of the running hook — a temporary
`MESSAGE` in `SUBCASE("the forward")`, rebuilt and run with `-s` — and the file
restored to its pristine sha256
`32ee46d64b3045cbf852c387a2cb106bc46d391ed9fd19b4b6cbc013afac9b07`, with `git
diff` empty on it.

## Mutation record — W5e-2 (#2336)

The wave that gave `Qwen4ExpTextPLELayer` a production composition,
`RunQwen4ExpPleBlock`, and with it the LAST of the three block seams. Measured on
base SHA `0bb090def` (`origin/main`) with W5e-1's branch `f249a854b` merged
forward, CPU only, Release, `-j 2`. The build return code was read BEFORE any
test result on every row, because under `ENOSPC` a failed build reads as a
passing test.

### The RED, before the change

The test file written first, against the tree with the two new source files
moved aside, compiling the test translation unit alone:

```
BUILD_RC=1
test_qwen4_exp_ple_block.cpp:70:10: fatal error:
  vllm/model_executor/models/qwen4_exp_ple_block.h: No such file or directory
compilation terminated.
```

That red is a BUILD refusal and is read as one, exactly as W5d-1's was: it says
the block is ABSENT, not that any arithmetic is wrong. The red for the intended
REASON is M1, M2 and M3, each of which builds clean and fails on values.

### No golden was added, and that is a decision

`scripts/gen-qwen4-exp-ple-goldens.py` already carried an end-to-end
`Qwen4ExpTextPLELayer.forward` (`kPleExpectedOutput`), its incremental twin, and
its masked arm (`kPleMaskedExpectedOutput`) — which is exactly what a block gate
needs, single-shot and across calls. Extending the generator would have been a
regeneration risk taken for nothing, so the generator and
`qwen4_exp_ple_goldens.inc` are BYTE-UNCHANGED by this wave. The oracle was
re-fetched and re-hashed independently: transformers `v5.16.0`
`models/qwen4_exp/modeling_qwen4_exp.py`, sha256
`77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`.

### The anchors, re-derived rather than relayed

Read line by line off the hashed file, because #2336's own gate citation was off
by one at BOTH ends and W5e-1 corrected it. `:1176` is the n-gram gather, `:1177`
`norm_key(key_proj(...))`, `:1178` `value_proj`, `:1179` `norm_query`, `:1180`
the dot and the `sqrt(hidden_size)` divide, `:1181` the signed square root,
`:1182` the broadcast sigmoid scale, `:1183` `norm_conv` over the FLATTENED
`gated_value`, `:1184` the flatten that becomes the skip term, `:1185-1187` the
two `apply_mask_to_padding_states` calls (`:204-213`), `:1188` the join with
`_short_conv` (`:1150-1167`). Two orderings matter and both were confirmed by
reading: `norm_conv` is applied to the flattened tensor, and the mask is applied
AFTER both norms and to BOTH tensors.

### What the FIXTURE can see, measured rather than asserted

Three cases run the W2 host reference with one defect injected and report the
distance from the oracle. They gate the FIXTURE and not the block: a golden on
which a defect is invisible is a mute switch, which is what #2272 recorded for an
eps probe that bound at two of four goldens. Each is the paired measurement for
one mutation below.

| Defect injected into the HOST reference | Separation from `kPleExpectedOutput` | tolerance |
|---|---|---|
| the n-gram history seeded with ZERO instead of `eos_token_id` | 1.2892 | 1e-5 |
| conv taps swapped, lag 9 against lag 6 | 2.27429 | 1e-5 |
| the `+ 1` dropped from the `norm_conv` gamma | 1.06722 | 1e-5 |
| the masked golden against the unmasked one (the mask is not inert) | 1.05815 | 1e-5 |

`T = 12` against a nine-column ring, so the ring both FILLS and is still being
rewritten at the last token; a fixture whose ring never wrapped could not tell
lag 9 from lag 6 at all, and the wrap is asserted rather than assumed.

### The f32 score accumulator at MODEL width, and how it is gated

W5e-1's `## Owed` recorded that `vt::BatchedMatmul` accumulates the `:1180` dot
in f32 with a sequential-over-K loop, and that in the clamp band an
accumulator-order difference can FLIP the gate's sign. That axis is this wave's
to gate, and it is gated ANALYTICALLY because a sign flip is a discrete outcome
that no tolerance bounds.

Measured at the released width, `hidden_size = 2560`, `hc_count = 4`, over an
adversarial fixture in which the `j == 0` pairs sum a large positive prefix and
subtract it back term for term (so the exact dot is zero while every partial sum
is O(H/2)) and the `j > 0` pairs are dense:

| quantity | measured |
|---|---|
| worst score disagreement against a double accumulator, in `g` units | 1.06371e-4 |
| near-null pairs, of 12 | 3 |
| sign flips among them | 2 |
| worst sigmoid delta, near-null | 1.48301e-3 |
| worst sigmoid delta, dense | 2.09894e-9 |

**THE FIRST DRAFT OF THIS CASE MEASURED NOTHING AND IS RECORDED BECAUSE OF IT.**
It built the near-null pairs as adjacent `+x, -x` terms, which cancel EXACTLY in
f32 as well as in double: the case reported a disagreement of zero, passed, and
would have passed with any accumulator at all. The prefix construction is what
makes the partial sums large enough to round.

**The bound is derived, not typed.** For two scores at most `dg` apart the worst
case is the pair `(+dg/2, -dg/2)` straddling the origin: the gate is then
`+/- sqrt(max(dg/2, 1e-6))` — the clamp is what puts the `max` there — and
`sigmoid` is 1/4-Lipschitz, so the outputs differ by at most
`0.5 * sqrt(max(dg/2, 1e-6))`. Away from the origin the square root is Lipschitz
with constant `1/(2 sqrt|g|)` and the difference is strictly smaller, so this is
a GLOBAL bound. At the stated `dg` bound of 1e-3 that is 1.118e-2, and the
measured 1.48301e-3 sits under it.

**THE HONEST CONCLUSION, WHICH IS NOT "THIS IS A DEFECT".** The near-origin
deviation is 1e2 to 1e3 times the suite's `kTol`, and no port of `:1181` can be
gated to 1e-5 there — ours or anyone's. Upstream's own line is discontinuous at
the origin, and upstream's own `:1180` is an f32 reduction too (torch's
`acc_type` for a float sum is float, with a pairwise blocking that is a different
ORDER rather than a wider accumulator), so two legitimate summation orders differ
here by construction. Mirroring vLLM means keeping the f32 accumulation, not
widening it; a double accumulator would be a divergence, and `vt::BatchedMatmul`
offers none in any case. **What follows for a token gate on this architecture is
recorded under `## Owed`:** a near-null PLE gate is a conditioning hazard of the
model, it is reached by ORDINARY rows and not only by padding, and the one place
it is harmless is harmless because of the MASK and not because of `value`.

**THE FIRST STATEMENT OF THIS WAS WRONG IN BOTH HALVES, AND THE FRESH REVIEW
MEASURED IT.** It said the band is produced in production only by a fully masked
row, "whose `value` is zero, so the gate scales zero either way". Neither clause
holds. `value` is `vt::MatmulBT(embeddings, value_proj)` at `:1178`, the n-gram
GATHER product, and nothing zeroes it on a masked row: the mask lands at
`:1185-1187`, AFTER the gate, as `vt::MulScalar(g_row, …, 0.0)` on the gated
output and `vt::MulScalar(n_row, …, 0.0)` on the conv input. The conclusion
survives and the mechanism does not — a masked row's gate output is DISCARDED,
whatever the gate returned and whatever `value` was.

**AND IT IS NOT THE ONLY CASE.** Sequential-over-K f32 (`cpu_ops.cpp:344-347`,
verbatim) against a double accumulator, over 2,000,000 random RMS-normalised
pairs at `hidden_size = 2560`, re-measured independently for this repair:

| quantity | measured |
|---|---|
| `rms(dg)`, in `g` units | 9.05e-7 |
| worst `dg` | 9.72e-6 |
| `\|g\| < 1e-6` | 4 of 2,000,000 |
| `\|g\| < 1e-3` | 1636 of 2,000,000 |
| sign flips | 1 |

**THE RATE IS THE NUMBER, NOT THE FLIP COUNT.** The review's own draw of the same
size reported `rms(dg) = 9.06e-7`, worst `1.12e-5`, 2 at `1e-6`, 1577 at `1e-3`
and ZERO flips. Two independent draws giving 1 and 0 are one Poisson rate with a
mean near 2, not a disagreement, so what is recorded here is the RATE: a sign
flip needs `|g|` below the `dg` scale, which is a per-score probability of order
1e-6. On the released one-PLE-layer config at `hc_count = 4` a 2048-token prefill
draws 8192 scores, so a NON-MASKED row reaches the band of order once per 1e2
prefills of that length. Padding is not what produces it.

**WHAT IS BOUNDED IS THE OUTPUT, PER hc BLOCK:** `0.5 * sqrt(max(dg/2, 1e-6))`
times `|value|`. At the typical `dg` the clamp floor dominates, because
`rms(dg) = 9.05e-7` is below the `2e-6` at which `dg/2` overtakes `1e-6`, and the
bound is 5.0e-4 * `|value|`; at the worst `dg` measured it is 1.1e-3 * `|value|`.

### The battery

Each mutation is a single edit, applied to a file whose pre-mutation sha256 is
recorded, built to a return code read BEFORE any test was run, and restored
byte-for-byte with `cp -a` followed by `touch` — `cp -a` preserves mtime and
ninja then SKIPS the rebuild. Every restore was proved by re-hashing.

Pre-mutation sha256:
`src/vllm/model_executor/models/qwen4_exp_ple_block.cpp` =
`dd89abd0b949870459c5891d3b6463e56ae336aadacd51b620060bdbfdb312bb`;
`src/vt/cpu/cpu_qwen4_exp_ple.cpp` =
`78cdbee1cbec809e616fe1bd113ae8511aa7b0708bca528c1e7d6a2d747f2242`.

| ID | Mutation | Build rc | Result |
|---|---|---|---|
| M1 | the conv kernel reads lag 6 where it owes lag 9 and lag 9 where it owes lag 6 (`cpu_qwen4_exp_ple.cpp`) | 0 | **RED**, 4 of 11 cases, 5 of 84 assertions. The oracle case, the incremental case, the bf16 case and both mask subcases |
| M2 | the block seeds the n-gram history with `0` instead of `p.eos_token_id` | 0 | **RED**, 4 of 11 cases, 5 of 84. This is the contract `## Owed` said nothing in this tree could see; it is seen now |
| M3 | `gemma = false` on the `norm_conv` grouped norm, i.e. the `+ 1` dropped (#2218's polarity) | 0 | **RED**, 4 of 11 cases, 5 of 84 |
| M4 | REACHABILITY — delete the production call site | n/a | **VACUOUS, NOT PASSING.** There is no production call site to delete. `grep -rln RunQwen4ExpPleBlock src include examples tests` returns the block's own header and body, the test, `tests/CMakeLists.txt`, and the refusal STRING in `qwen4_exp_registry.cpp` — which names the symbol in prose and does not call it. Said in the commit body, the pull-request body and `## Owed` |
| M5 | the block passes `dilation = 1` to `vt::Qwen4ExpPleConv` | 0 | **RED**, 4 of 11 cases, and by a NAMED REFUSAL rather than by values: "qwen4_exp_ple_conv: conv_state must be [N,C,(K-1)*dilation] = [N,16,3], got [N,16,9]". The op's own cross-check is what makes a wrong dilation unreachable from this block |
| M6 | mask `gated_value` only, leaving `gated_value_normed` unmasked (upstream masks BOTH at :1186-1187) | 0 | **RED**, 1 of 11 cases, 2 of 84 — and the SMALL count is the finding: only the mask case moves, which is the localization the fixture claims |
| M7 | delete the `conv_mask` PAIRED-obligation refusal | 0 | **RED**, 1 of 11 cases, 1 of 84: the refusal case reports "did NOT throw at all". The `## Owed` entry it discharges is therefore closed by an assertion and not by a paragraph |
| M8 | `gate_divisor = 1.0f`, dropping the `sqrt(hidden_size)` tail of :1180 | 0 | **RED**, 4 of 11 cases, 5 of 84 — including the bf16 case, but only after its bound was tightened; see below |

**M8 IS THE REASON THE bf16 BOUND IS 0.05 AND NOT 0.2.** At the first bound the
bf16 case survived M8, so it proved that bf16 bytes FLOW and nothing about their
values. The two numbers were then measured — 1.007e-2 clean, 9.171e-2 under M8 —
and the bound moved between them, so the bf16 arm now convicts the same defect
population the f32 arm does. The tightening was made before the battery was
re-run, and every row above is measured on the FINAL head.

### Counts, before and after, on the same tree

`git diff --numstat` over `src/` and `include/` is `35/11` on exactly one file,
`qwen4_exp_registry.cpp`, and that file's only functional change is the refusal
STRING; everything else this wave adds is two new files. So every suite below
except the new one is expected unchanged, and the two that drive the refusal
string are the check on that expectation.

| suite | at `f249a854b` | at this head |
|---|---|---|
| `test_qwen4_exp_ple_block` | did not exist | 11 / 84 / rc 0 |
| `test_qwen4_exp_ple` | 9 / 395 / rc 0 | 9 / 395 / rc 0 |
| `test_qwen4_exp_ple_device` | 10 / 538 / rc 0 | 10 / 538 / rc 0 |
| `test_qwen4_exp_ple_gate` | 9 / 176 / rc 0 | 9 / 176 / rc 0 |
| `test_qwen4_exp_scaffold` | 12 / 296 / rc 0 | 12 / 296 / rc 0 |
| `test_qwen4_exp_forward` | 1 / 421 / rc 0 | 1 / 421 / rc 0 |
| `test_qwen4_exp_qsa_block` | 11 / 4382 / rc 0 | 11 / 4382 / rc 0 |
| `test_qwen4_exp_moe` | 5 / 112 / rc 0 | 5 / 112 / rc 0 |
| `test_qwen4_exp_kv_cache` | 4 / 399 / rc 0 | 4 / 399 / rc 0 |
| `test_ops_rms_norm_group` | 7 / 69 / rc 0 | 7 / 69 / rc 0 |

### The refusal string was falsified by this change, and repaired in it

W5e-1 wrote "there is no RunQwen4ExpPleBlock beside RunQwen4ExpQsaBlock and
RunQwen4ExpMoeBlock … and it is the LAST one", deliberately naming the SYMBOL so
that a reader could grep for it and so that the sentence would resolve the day
this wave landed. It resolves here, so the clause is REMOVED rather than
reworded — a refusal enumerates what is missing and a present item is not
missing — and the "LAST one" sentence goes with it, because the population it
counted is empty.

`test_qwen4_exp_scaffold.cpp` pins five substrings as PRESENT and cannot pin any
of them TRUE, so the repair was verified by READING THE EMITTED BYTES out of the
running hook: a temporary `MESSAGE` in `SUBCASE("the forward")`, rebuilt, run
with `-s`, then the file restored to its pristine sha256
`32ee46d64b3045cbf852c387a2cb106bc46d391ed9fd19b4b6cbc013afac9b07` with `git
diff` empty on it. The bytes were read TWICE, because the first reading found a
second defect worth repairing: the string said "is now on main" while sitting on
a branch whose own op was not on main, so it was false at the commit that
authored it and true only after a merge nobody had made. It now says "is now in
this tree", which is checkable at every commit.

### What the battery did NOT reach

A battery's silence is not a result.

- **No production entry point reaches this block**, so M4 measures nothing; that
  is stated as VACUOUS above rather than counted as a pass.
- **No CUDA arm exists** for any op this block is the first production caller of,
  so none can be mutated, and the block's host round trip for the n-gram hash has
  no device form to compare against.
- **No block-quantized weight is exercised.** `LoadMatmul` can hand `key_proj`
  and `value_proj` to the block as keep-quant blocks and `vt::MatmulBT`
  auto-dispatches `kMatmulBTQuant` on them, but the goldens are f32 and bf16
  only, so the quantized arm of the two projections is reasoned about and not
  measured. The n-gram TABLE's keep-quant arm is likewise unexercised here;
  W6a gates it directly.
- **`num_reqs > 1` is out of reach**, as it is for `RunQwen4ExpQsaBlockPaged`:
  `vt::Qwen4ExpPleConv` is batched but the n-gram history is per sequence and
  `BuildNGramIds` takes one stream of ids.
- **TWO REFUSALS LAND IN THIS WAVE AND ONLY ONE OF THEM IS MUTATED. THE BRIEF
  THAT PRODUCED THE WAVE CONFLATED THEM, AND THE FRESH REVIEW SPLIT THEM.**
  **M7 covers the `conv_mask` PAIRED-EOS refusal** and nothing else: it deletes
  that refusal from `qwen4_exp_ple_block.cpp` and the suite reds with "did NOT
  throw at all". **`Qwen4ExpPleLayout`'s STATED-versus-DERIVED `head_vocab_sizes`
  refusal is a different refusal, and it is covered by DIRECT THROW-TESTS WITH NO
  MUTATION** — the three `CHECK_THROWS_WITH_AS` subcases of "Qwen4ExpPleLayout
  refuses STATED head vocabulary sizes that disagree", which assert the thrown
  type and the message text against a disagreeing size and against a short set.
  A direct throw-test is weaker evidence than a mutation, because it proves the
  refusal fires and not that deleting it would be caught anywhere else; it is
  recorded as what it is rather than folded into the M-numbered battery.
- **No token, no speed, no device.** CPU only, and nothing decodes.

## Mutation record — W5f (#2031, #2336)

`Qwen4ExpTextModel::Forward` — the layer loop, and the first production forward
this architecture has had. Base: `f060a81d6` (W5e-2, `row/MODEL-MM-QWEN4-EXP-W5E2`),
which is **not on `main`** — GitHub write access was suspended (HTTP 403) for the
whole of this wave, so nothing could be pushed and this branch is stacked on
W5e-2 deliberately.

### The oracle, and that it is STANDING

transformers **5.16.0**, the accepted lane pin, INSTALLED and RUNNING rather than
read: `scripts/gen-qwen4-exp-forward-goldens.py` imports `Qwen4ExpTextModel`,
asserts `sha256(modeling_qwen4_exp.py) ==
77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459` against the
file it imported, seeds every parameter from a bf16-EXACT deterministic grid
(asserted per tensor with a `bfloat16` round trip), and calls `forward`. The
lane's `gateable = no` is about the RELEASED CHECKPOINT and stays as it is; a
tiny random config is a different question and it runs on CPU in seconds.

### The anchors, re-derived rather than relayed

Read out of that same file. `#2336`'s cited range was off by one at both ends
once already, so every line below was located by reading, not by citation.

| Anchor | Line | What it says |
|---|---|---|
| `Qwen4ExpTextModel.forward` | `:1415` | `hidden_states = inputs_embeds` |
| | `:1416` | `position_embeddings = self.rotary_emb(hidden_states, position_ids)` |
| | `:1417` | `hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)` |
| | `:1419` | `for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers])` |
| | `:1430` | `hidden_states = self.hyper_connection_mixer(hidden_states)` |
| `Qwen4ExpTextDecoderLayer.forward` | `:1218` | `hidden_states = hidden_states + self.ple(...)` — **FIRST in the layer** |
| | `:1222` | `hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)` |
| | `:1224` | `hidden_states = self.linear_attn(...)` |
| | `:1228` | `hidden_states, _ = self.self_attn(...)` |
| | `:1236` | `injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)` |
| | `:1237` | `hidden_states = hyper_input + injection.flatten(-2)` |
| | `:1239` | the MLP hyper-connection, identical shape to `:1222` |
| | `:1240` | `hidden_states = self.mlp(hidden_states)` |
| | `:1242-1243` | the second injection and write-back, identical to `:1236-1237` |
| `Qwen4ExpTextDecoderLayer.__init__` | `:1202` | `ple_layer_index = config.ple_layer_ids.index(layer_idx + 1)` — ONE-BASED |
| `Qwen4ExpTextExperts.forward` | `:889` | `linear(x, gate_up_proj[e]).chunk(2, dim=-1)` — the gate half is the FIRST `I` ROWS |

**AND ONE ANCHOR THAT IS A DIFF RATHER THAN A LINE.**
`Qwen4ExpTextGatedDeltaNet` (`:403-564`) and `Qwen3_5GatedDeltaNet`
(`modeling_qwen3_5.py:387-547`) are byte-identical class bodies except for one
hunk: `RMSNormGated(..., activation=config.output_gate_type or
config.hidden_act)` against `RMSNormGated(head_v_dim, eps=...)`. That is the
whole justification for routing 36 of 48 layers through `RunGdnBlockPaged`
instead of writing a second Gated DeltaNet, and it was measured by diffing the
two classes rather than argued.

### The RED, before the change

The loop was written before the golden ran, so the first RED is the first run of
the gate against it rather than a pre-implementation failure, and it is recorded
as what it is. It was not a formality — it found two real defects, one in the
fixture and one in this record's own conditioning:

```
tests/vllm/models/test_qwen4_exp_layer_loop.cpp:512: MESSAGE: layer loop vs
  transformers 5.16.0: max|diff| = 0.466067 against a bound of 0.03
tests/vllm/models/test_qwen4_exp_layer_loop.cpp:514: ERROR: CHECK( worst < kTol )
  is NOT correct!  values: CHECK( 0.466067 <  0.03 )
[doctest] test cases:  1 |  0 passed | 1 failed | 0 skipped
[doctest] assertions: 65 | 64 passed | 1 failed |
```

Bisected with a temporary per-stage probe against oracle forward hooks, which
put the defect at one stage rather than leaving 0.466 to be argued about:

| stage | max\|diff\| |
|---|---|
| layer 0 in (embed + widen) | 0 |
| attn hyper-connection `mixed` | 0.00173 |
| attn hyper-connection `injection` | 0.00362 |
| Gated DeltaNet output | 0.00200 |
| MLP hyper-connection `mixed` | 0.00586 |
| **MoE block output** | **0.20813** |
| layer 0 out | 0.24583 |

**DEFECT 1, IN THE FIXTURE: `nk`.** `LoadMatmul` (`qwen4_exp_weights.cpp`) sets
`nk = true` on EVERY arm, and the suite's `Bf16` helper defaulted to `false`. It
does not matter for a weight the consumer reads through
`dense_attn::ResidentWeight` + an explicit shape, which is most of them; it
matters for the MoE shared expert, where `BorrowWhole` PRESERVES the source's
flag and hands it to `MatmulF32D`. Same bytes, same element count, no shape
error, read in the wrong orientation. The helper's default is now `true`, stated
once with the reason rather than at thirty call sites.

**DEFECT 2, IN THE FIXTURE'S CONDITIONING: a top-k router on the boundary.** With
`nk` fixed the MoE still read 0.208, and the cause was not arithmetic. The first
draft used two experts at top-1 and a uniform-random router over an 8-wide
hidden; its worst logit margin was **0.0164** against a hidden-state residual of
**0.0152**. A top-k selection has BIMODAL error — the two sides pick the same
experts and the residual is bf16-sized, or they pick different ones and it is
O(1) — so no tolerance can straddle it. Fixed by CONDITIONING and not by
widening: four experts at top-2 (which also keeps `norm_topk_prob` observable,
where top-1 renormalizes to exactly 1.0 whatever the logits are), a `ROUTER_SCALE`
of 8 applied to `mlp.gate.weight` and nothing else, and a `SALT` chosen by
MEASURING the worst margin over 20 draws. Worst margin **0.534569502**, emitted
into the golden, and the suite REQUIRES it stays above 0.25 so a regeneration
that drifts back onto the boundary fails loudly instead of reporting a large
residual that reads like a broken loop.

### The result

```
tests/vllm/models/test_qwen4_exp_layer_loop.cpp:558: MESSAGE: layer loop vs
  transformers 5.16.0: max|diff| = 0.00982457 against a bound of 0.03
[doctest] test cases:  2 |  2 passed | 0 failed | 0 skipped
[doctest] assertions: 83 | 83 passed | 0 failed |
```

The assertion count is 83 and not the 76 this section first recorded, because the
review repair below replaced one bare `CHECK_THROWS` with eight message
assertions across two refusals. The measurement itself is unchanged at
`0.00982457`, which is the check that the repair touched the suite and not the
arithmetic.

**THE BOUND IS 3.0e-2 AND THE MEASUREMENT IS 9.8e-3, and the gap is stated rather
than hidden.** The oracle runs the tower in f32 and this tree runs the model path
in bf16, which AGENTS.md "Inherit vLLM defaults" requires; the WEIGHTS are
bf16-exact by construction, so the residual is activation rounding over four
layers and nothing else. A bound three times the measurement is not a mute
switch — it is above the real value, not below it — and what makes it a gate
rather than a number is the separation below. **SIX of the ten rows below carry
a `max|diff|`** — M1b, M2, M3, M4, M6c and M7 — and every one of those six lands
between 0.78 and 2.02, which is 26x to 69x the bound and 79x to 206x the
measurement. The word "every" used to stand here without the count and it was an
overstatement: the other four rows say something a `max|diff|` cannot. M1 reds on
the loop's OWN guard and never reaches the comparison, so it emits no number;
M5 is the reachability split, whose result is WHICH assertions red rather than
how far a value moved; M6 reds on a direct `REQUIRE` over a string; and M6b was
WITHDRAWN at build rc 1 and is not a test result at all.

### The battery

Each mutation: sha256 before and after (they must DIFFER, which proves it
applied), the BUILD rc read BEFORE any test result, the run, then a restore
proved byte-identical by sha256 with a `touch` after it so ninja cannot skip the
rebuild.

| # | Mutation | Build | Result |
|---|---|---|---|
| M1 | the layer-kind predicate is inverted, so a `linear_attention` layer takes the sparse arm | rc 0 | **RED** — but on the loop's OWN guard (`lw.is_linear_attention == linear`), not on the golden, so it does not prove the golden sees layer order |
| M1b | the same property where that guard is blind: the stack is walked in REVERSE, `lw` and `p.layer_types[il]` still agreeing at every step | rc 0 | **RED**, max\|diff\| **1.30131** |
| M2 | the attention hyper-connection's rank-1 write-back (`:1236-1237`) is dropped | rc 0 | **RED**, max\|diff\| **1.28974** |
| M3 | the n-gram history is seeded with ZERO instead of `eos_token_id` (a VALID token id, so nothing crashes) | rc 0 | **RED**, max\|diff\| **0.777988** |
| M4 | the PLE block runs LAST in its decoder layer instead of first (`:1218`) | rc 0 | **RED**, max\|diff\| **1.03257** |
| M5 | **reachability**: the `Qwen4ExpTextModelForward` call site is deleted from the registry hook AND the pre-W5f unconditional refusal string is restored in its place | rc 0 | **RED** on 3 assertions of the reachability case — and the golden case stays GREEN at 0.00982457, which is the separation `.agents/reachability.md` step 5 exists to produce |
| M5-bare | the same call site deleted and NOTHING put back: `hidden` becomes an uninitialised `[T, hidden_size]` buffer and the rest of the hook survives | rc 0 | **RED on ONE FATAL assertion.** The hook returns logits, so the case's `FAIL(...)` fires at `test_qwen4_exp_layer_loop.cpp:716` and doctest ABORTS the case: `1 of 2 cases`, `72 of 83 assertions` reached, and the golden still reads `0.00982457`. Recorded separately because M5's three reds come from a TWO-PART construction and only this row measures the deletion on its own |
| M6 | `Qwen4ExpGdnHfConfig` stops CARRYING `output_gate_type` and takes the shared-reader default | rc 0 | **RED**, on the direct `REQUIRE` — a transcription check, so M6c follows |
| M6b | `GdnSigmoidGate` returns `false` | **rc 1** | **WITHDRAWN.** `-Werror=unused-parameter`: `cfg` became unused. A build failure is not a test result |
| M6c | M6b's replacement: `GdnSigmoidGate` INVERTS its predicate, so `cfg` is still read and the config still SAYS sigmoid | rc 0 | **RED**, max\|diff\| **1.37979** — the golden sees the GDN output-gate ACTIVATION, which is [#489](https://github.com/mudler/vllm.cpp/issues/489)'s axis and had never been gated on this architecture |
| M7 | the widen becomes a TILE (`idx[i] = i % T`) instead of a REPEAT (`idx[i] = i / hc`) — same shape, same multiset of values | rc 0 | **RED**, max\|diff\| **2.02334** |

**M1 AND M5 ARE THE TWO THAT SAY SOMETHING THE OTHERS CANNOT.** M1 reds on the
wrong instrument and is recorded as such rather than counted as a pass for the
golden; M1b is what actually gates layer order. M5 is the first NON-VACUOUS
reachability mutation this row has ever run — W5b-5, W5d-3, W5d-4, W5e-1 and
W5e-2 each recorded it as vacuous because there was no production call site to
delete.

### The refusal string was falsified by this change, and repaired in it

For the EIGHTH time on this row (#2288). The pre-W5f refusal said "the forward is
not ported yet ... What is missing is the LAYER LOOP itself,
`Qwen4ExpTextModel::Forward`". This wave writes it, so the whole string is
removed rather than reworded, and what replaces it are the two refusals that are
TRUE at this commit — `past_len != 0` and `num_reqs != 1` — each naming the
engine seam that owes it. **The emitted bytes were read out of the running hook,
not grepped**, with a temporary `MESSAGE` that was removed and the file proved
restored by sha256 (`0e2990d6728085447f0efe1f4a119f93eeb2862ea0febe157a6316290630e125`
before and after):

```
vt: Qwen4ExpForConditionalGeneration: this forward serves a SINGLE-SHOT PREFILL
(past_len == 0) and this step continues a sequence at past_len 1. ... The channel
that would carry them is multi_kv, which ModelRegistry::Forward refuses by name
and which #2353 established must not be lifted until a consuming forward and a
recurrent-member channel exist. Owned by KV-DSV4-MULTICACHE W5 (#1925, #2068)
and #2353 ...
```

**AND `test_qwen4_exp_scaffold.cpp`'s `SUBCASE("the forward")` INVERTED, which is
a change of meaning and not a weakening.** It asserted that the message said
"forward is not ported", that it named W2 and W4, and that it was NOT the
type-mismatch report — because the refusal had to precede the `ModelAs` downcast
or be unreachable. That argument had one premise, stated in the registry TU
itself: "nothing can produce a loaded Qwen4-Exp while the loader refuses". W5a
made the loader LOAD and W5f made the forward RUN, and the same comment named
this as the moment to restore the house ordering. The downcast is now FIRST, and
the bytes a foreign handle gets were read out of the running hook too:

```
Qwen4ExpForConditionalGeneration: the LoadedModel handed to this registry entry
point was not produced by Qwen4ExpForConditionalGeneration's own load_weights
... Refusing by name rather than downcasting a foreign model, which is undefined
behaviour on every member call that follows (issue #775).
```

### Counts, before and after, on the same tree

| Suite | Before | After |
|---|---|---|
| `test_qwen4_exp_layer_loop` | did not exist | 2 cases / 83 assertions |
| `test_qwen4_exp_scaffold` | 12 / 296 | 12 / 294 |
| `test_qwen4_exp_forward` | 1 / 421 | 1 / 421 |
| `test_qwen4_exp_ple_block` | 11 / 84 | 11 / 84 |

The scaffold's two-assertion drop is the rewritten `SUBCASE("the forward")`: six
checks became four, because the enumeration of owing waves is gone with the
refusal that carried it.

### What the battery did NOT reach

Stated so a reader does not infer coverage from ten reds. No mutation of the QSA
arm reds anything here that `test_qwen4_exp_qsa_block.cpp` does not already
gate — the loop's QSA layer is one of four and its own suite is stronger on it.
Nothing drives a quantized arm, a CUDA arm, a masked prefill or the released
48-layer geometry; those are `## Owed`. A step at `num_reqs = 2` IS driven, but
only as far as the hook's refusal — no multi-request batch is computed, and the
refusal is the whole of what that case proves. And nothing decodes a token, on
any hardware, which is the sentence this row has to keep writing until the engine
seams land.

## Stop conditions

- vLLM registers `qwen4_exp`: **stop and reconcile onto vLLM** before continuing.
  This is the designed end of the transformers exception.
- SGLang #36497 merges: re-survey the op mapping; it does not displace vLLM.
- The transformers lane pin is rejected in review: the row holds at `READY` and the
  gate stays `PENDING`. Do not proceed on an unpinned oracle.
- No arm is made to fit any fleet device: the row holds with G0 passed and G1-G3
  `PENDING` on hardware, recorded as visible debt, and no token claim is made.

## The refusal boundary

W1's whole product is a boundary: which configs this port accepts and which it
refuses. This row has **no reachable token gate** (`## Gates`, `gateable = no`), so
nothing downstream will ever catch a wrong default by running the model — a
`partial_rotary_factor` read from the wrong place, an n-gram field defaulted to
zero, or a missing `eos_token_id` all produce a config that parses, resolves, and
is silently wrong for W2 and W4. The config layer is the last place any of it is
checkable, so the boundary is **measured** here rather than described.

**The oracle runs.** `transformers` 5.16.0 installs and imports without torch —
it says so itself ("only tokenizers, configuration and file/data utilities can be
used") — and `Qwen4ExpConfig.from_dict()` runs `Qwen4ExpTextConfig.__post_init__`
and `validate_architecture` in full. That makes the CONFIG layer of this row
gateable even though the MODEL layer is not, and it is the only layer of this row
that is. `gateable = no` in `oracles/transformers.md` still stands: it is a
statement about running the model, and nothing here runs one.

### Two-direction sweep

39 configs, each derived from the committed fixture, put through
`Qwen4ExpConfig.from_dict` on one side and `LoadHfConfig -> ModelRegistry::Resolve
-> factory->parse_config -> ParseQwen4ExpParams` on the other. **35 agree; 4
differ, and over these 39 all 4 are ours refusing what upstream accepts** — the
safe direction, since the reverse is what lets a bad checkpoint through.

**That is a claim about the measured set, and it is bounded on purpose.** An earlier
draft said "never the reverse" as an absolute, and a fresh re-review falsified it with
a fortieth case outside the sweep: `rope_parameters` carrying **`rope_dim = 64`
alongside `partial_rotary_factor = 1.0`**. Upstream ignores `rope_dim` entirely —
`validate_architecture` computes `int(self.head_dim * partial_rotary_factor)` = 256
unconditionally at `configuration_qwen4_exp.py:225-226` — and refuses, because
256 > `indexer_head_dim` 128. We take `rope_dim` in preference, following vLLM's
`get_rope` semantics in the shared reader (`hf_config.cpp:545-547`), and **ACCEPT at
`rotary_dim = 64`**, handing W4 a 64-of-256 slice. That is the same failure mode and
the same direction as the finding that failed this wave's first review, reached
through a different key.

It is narrow and it is not a defect in this model's code: `rope_dim` has **zero
occurrences** in `modeling_rope_utils.py` at v5.16.0, so no transformers path writes
or reads it and no published checkpoint carries it — the oracle tolerates the key and
ignores it. The divergence lives in the shared reader, which is deliberately mirroring
vLLM rather than transformers on that point.

It is recorded rather than repaired because the fix belongs to whoever reconciles the
shared reader's rope resolution, not to this row, and because the honest form of a
boundary claim in the row whose whole product is that boundary is either **true or
bounded**. Owed: either a `rope_dim` case in the sweep with the divergence stated, or
a shared-reader change that makes it moot.

Reproduce (transformers 5.16.0 in a venv; the probe links `build/libvllm.a` with
`-Wl,--whole-archive` so the model's self-registration survives):

| upstream verdict | ours | cases |
|---|---|---|
| ACCEPT | ACCEPT | baseline; `prf` top 1.0 / rope .25; `prf` top .25 / rope absent; `eos_token_id` null with PLE OFF; every n-gram default omitted; no `output_gate_type` with `hidden_act` silu; all five indexer keys erased; `layer_types` erased (interval synthesis) |
| REFUSE | REFUSE | `prf` absent everywhere; `prf` only in rope 1.0; `prf` top .25 / rope 1.0; `eos_token_id` null with PLE ON; `eos_token_id` `[]`; no `output_gate_type` with `hidden_act` gelu; `output_gate_type` swish; `output_gate_type` gelu; `ple_embed_dim` -2560; `ple_embed_dim` 2561; `hc_count` 1; `num_experts` 0; `num_experts_per_tok` 513; `moe_intermediate_size` 0; partial QSA group; `indexer_n_heads` 0; `indexer_kv_heads` 2; `indexer_budget` 2049; `sliding_attention`; `ple_layer_ids` [0]/[4]/[49]; `ngram_size` 1; `heads_per_ngram` 0; interval 0; short `layer_types`; `num_hidden_layers` 0 |
| ACCEPT | **REFUSE** | `hc_lowrank` 0; `ple_conv_kernel_size` 0; `mtp_num_hidden_layers` -1; `partial_rotary_factor` -0.25 |

### Each upstream rejection, and the line that implements it

`configuration_qwen4_exp.py` at `v5.16.0`; local lines in
`src/vllm/model_executor/models/qwen4_exp.cpp` unless stated.

| # | upstream | our implementation | exercised by |
|---|---|---|---|
| 1 | `:190-192` unsupported `layer_types` | `KindFromString` | "an unsupported layer type" |
| 2 | `:193-195` `output_gate_type or hidden_act` not in {sigmoid, silu} | the raw-text gate resolution, NOT `config.output_gate_type` | "[UP] an absent output_gate_type falls back to hidden_act", "[UP] an explicit output_gate_type outside {sigmoid, silu}" |
| 3 | `:196-197` `hc_count <= 1` | the `hc_count` refusal | "hc_count must exceed 1" |
| 4 | `:198-199` `num_experts <= 0` | the `num_experts` refusal | "[UP] num_experts must be positive" |
| 5 | `:200-204` `num_experts_per_tok` outside [1, num_experts] | the `num_experts_per_tok` refusal | "num_experts_per_tok above num_experts" |
| 6 | `:205-206` MoE intermediate sizes | the MoE-size refusal | "[UP] the MoE intermediate sizes must be positive" |
| 7 | `:216-218` partial QSA group | the `present != 5` refusal, naming the missing fields | "a partial QSA group names what is missing" |
| 8 | `:219-220` QSA values not positive | the QSA positivity refusal | "[UP] QSA values must be positive" |
| 9 | `:221-222` `indexer_kv_heads != 1` | the `kv_heads` refusal | "QSA requires exactly one indexer kv head" |
| 10 | `:223-224` `indexer_budget % indexer_compress_ratio` | the divisibility refusal | "the indexer budget must divide by the compress ratio" |
| 11 | `:225-231` `rotary_dim > indexer_head_dim` | the refusal, over `config.rotary_dim` from the SHARED reader | "absent everywhere: 1.0, rotary_dim 256, and upstream REFUSES", "top-level 0.25 does NOT rescue a rope dict that says 1.0" |
| 12 | `:235-239` `ngram_heads <= 0 or ple_embed_dim <= 0 or ple_embed_dim % ngram_heads` | split three ways so the message names the field: `ngram_size < 2`, `heads_per_ngram <= 0`, then `heads <= 0 \|\| embed_dim <= 0 \|\| embed_dim % heads` | "[LOCAL] ngram_size below 2", "[LOCAL] heads_per_ngram must be positive", "[UP] a NEGATIVE ple_embed_dim", "[UP] a ple_embed_dim that does not divide by the head count" |
| 13 | `:240-247` `ple_layer_ids` outside [1, num_hidden_layers] | the one-indexed range refusal | "a PLE id outside the one-indexed range" |
| 14 | `:248-255` PLE on a non-`linear_attention` layer | the layer-kind refusal | "a PLE id on a sparse-attention layer" |
| 15 | `:256-257` `eos_token_id` unset with PLE enabled | the `eos_token_id` refusal | "[UP] eos_token_id must be set when PLE is enabled", "[UP] an EMPTY eos_token_id list is refused too" |

`__post_init__` behaviors, which are not rejections but decide what the rejections
see: `full_attention -> qwen_sparse_attention` (`:180-184`), the interval synthesis
(`:174-179`), `ple_embed_dim` defaulting to `hidden_size` (`:168`),
`sorted(set(ple_layer_ids))` (`:167`), and `number_of_conv_states` (`:172`). Each
has its own case.

**Upstream's ORDER inside the PLE block is mirrored**, and deliberately: head count
and embedding width first, then the id range, then the layer kind, then EOS. A
config violating two at once has to report the one upstream reports, or a reader
comparing the two runtimes is sent to a different field.

### Every refusal is mutated ONE AT A TIME

A sweep is an accept/reject comparison; it does not say whether OUR TESTS would
notice a refusal going missing. So each of the 23 refusals in
`ParseQwen4ExpParams` was deleted individually — `if (<guard>) {` rewritten to
`if (false) {`, proved applied by a non-empty `git diff --stat`, rebuilt, run,
and restored by byte comparison. **All 23 red.** Before this change a single
mutation deleting 13 of them at once left the suite green.

Deleting them as a UNION is not equivalent and would have hidden two defects: the
first union mutation SIGFPE'd on `(i + 1) % 0` at the second subcase and never
reached the other eleven. Run one at a time, two of the new subcases turned out
to be weak — `num_hidden_layers = 0` asserted the bare field name, which the next
refusal down ("`layer_types` has 48 entries but `num_hidden_layers` is 0") also
prints, and `num_experts = 0` the same against the `num_experts_per_tok` range
message. Both now assert the distinguishing text. That is the general shape:
**a substring assertion is a weak gate wherever two refusals share a word**, and
only a per-guard mutation finds it.

The three production entry points were mutated too. Gutting the registered
`parse_config` hook to `(void)config;` reds 3 cases / 42 assertions; removing the
forward's `VT_CHECK` reds 5 assertions; removing the GGUF arm's throw reds 4.
Before this change all three were green.

**W5f REPLACED THAT FORWARD `VT_CHECK`, AND ITS FIRST HEAD SHIPPED THE TWO
REPLACEMENTS UNGATED. Fresh review caught it and this is the repair.** The
mutation-proven unconditional refusal is gone; what stands in its place are two
narrower ones, `past_len != 0` and `num_reqs != 1`. As landed, neither was
detected by anything:

| # | Mutation, applied ONE AT A TIME | Build | As landed | After the repair |
|---|---|---|---|---|
| MR1 | the `past_len == 0` `VT_CHECK` is deleted from `ForwardQwen4ExpForConditionalGeneration` | rc 0 | **GREEN.** `test_qwen4_exp_layer_loop` 2 cases / 76 assertions SUCCESS | **RED, 4 of 83 assertions**, rc 1 |
| MR2 | the `num_reqs == 1` `VT_CHECK` is deleted from the same hook | rc 0 | **GREEN.** layer_loop 2 / 76 AND `test_qwen4_exp_scaffold` 12 / 294, both SUCCESS | **RED, 4 of 83 assertions**, rc 1 (the scaffold suite stays green, correctly: a foreign handle cannot reach past the downcast) |

**WHY IT PASSED, AND IT IS THE `## Testing traps` shape rather than an
oversight.** The only assertion was a bare
`CHECK_THROWS(ModelRegistry::Forward(*model, in2))`. Delete the `past_len` guard
and that input does not sail through — it runs INTO the loop and throws at layer
1's PLE layout cross-check, the same exception the reachability case above lands
on. An unrelated throw satisfied the assertion, so the guard was a MUTE SWITCH.
`num_reqs != 1` was worse: no test constructed a step with more than one request,
so nothing drove it at all.

**The repair asserts each refusal TWO-SIDED on its own MESSAGE**, which is the
same lesson the 23-guard sweep above already recorded ("a substring assertion is
a weak gate wherever two refusals share a word"). For each: the bytes that
identify the refusal are PRESENT, including the VALUE it reports back
(`at past_len 1`, `the step carries 2`), AND `qwen4_exp ple layout` is ABSENT,
which is what proves the input stopped at the boundary instead of entering the
loop. Deleting either guard flips both halves of its pair, which is why MR1 and
MR2 now red on four assertions each rather than passing.

### Refusals we impose that upstream does not

Each is deliberate, each is exercised, and each is a row in the sweep above. None
of them lets a config through that upstream refuses.

| ours | upstream | why we keep it |
|---|---|---|
| `num_hidden_layers <= 0` | none | a zero-layer stack is unrepresentable downstream; upstream refuses the same fixture for a different reason (the PLE id range collapses to [1, 0]) |
| `layer_types` length vs `num_hidden_layers` | none | upstream indexes `layer_types[layer_id - 1]` and would `IndexError`; in C++ that is an out-of-bounds read |
| `full_attention_interval <= 0` | none | `(i + 1) % 0` is UB in C++ where Python raises `ZeroDivisionError` |
| `hc_lowrank <= 0` | none | a non-positive rank cannot size the hyper-connection mixer W3 builds |
| `ple_conv_kernel_size <= 0` | none | `short_conv_state_len()` goes negative and W2 sizes a conv state from it |
| `mtp_num_hidden_layers < 0` | none (not even a declared field of `Qwen4ExpTextConfig`) | a negative depth cannot be built |
| `ngram_size < 2` / `heads_per_ngram <= 0` | folded into `ngram_heads <= 0` | same accept/reject boundary, a message that names the field |
| non-integer / non-array JSON where a number or list belongs | Python coerces or raises later | a typed reader has to refuse at the boundary |
| `partial_rotary_factor` outside (0, 1] | none | **belongs to the SHARED reader**, `hf_config.cpp`, not to this model. It fires before this parse runs, which is why there is no local guard: one would be unreachable. Recorded here because the sweep sees it as ours |

### What the config layer still cannot see

`Qwen4ExpParams` resolves the fields W1 through W5 consume. It does NOT yet carry
`linear_num_key_heads` (16), `linear_num_value_heads` (**48**, against upstream's
declared default of 32), `linear_key_head_dim`, `linear_value_head_dim`,
`linear_conv_kernel_dim`, `norm_topk_prob`, `max_position_embeddings` or the
resolved `output_gate_type` value. The shared reader types most of them, so
nothing is lost — but a wave titled "config resolution" owes the statement, and it
is listed under `## Owed`.

## The PLE layout's two sources, and which one is the authority (W5g, #2031)

**A TOKEN CAME OUT.** On 2026-08-30, on `row/MODEL-MM-QWEN4-EXP-E2E` (W5e-2 + W5f
+ `ENG-MULTIKV-FORWARD-1925` + `ENG-MULTIKV-BYNAME` composed onto `origin/main`
at `7d53ae3b4`), `ModelRegistry::Forward` ran a complete single-shot prefill on a
model loaded by `ModelRegistry::Load` from a synthetic `qwen4exp` GGUF, returned
`[1, 16]` f32 logits with every element finite, and `vt::GreedyArgmax` sampled
token id **15**. Logit range `[3290.84, 95090.7]`. CPU, no GPU, no released
checkpoint. `tests/vllm/models/test_qwen4_exp_layer_loop.cpp`, case
`ModelRegistry::Forward reaches it on a loaded qwen4exp GGUF`.

That is a REACH and a SAMPLE, not a token gate. The fixture's weights are a
deterministic ramp, so no reference token stream exists and id 15 is compared
against nothing. The tower's ARITHMETIC is gated separately against the
lane-pinned transformers 5.16.0 oracle, unchanged by this wave at
`max|diff| = 0.00982457` against a 3.0e-2 bound.

### What stood in the way, and why it was not only the fixture

W5f recorded the blocker as a fixture defect. **It is a port defect that the
fixture was the first thing to expose**, and the correction matters because the
fixture repairs W5e-2 proposed would each have left the port defect standing.

The n-gram head vocabulary has two possible sources and they never coexist:

| source | states `ngram_vocab_size_base` | states `ple.head_vocab_sizes` / `ple.head_offsets` |
|---|---|---|
| `config.json` | yes | no — HF derives them |
| `qwen4exp` GGUF | **no** | yes — llama.cpp #27742 writes the resolved arrays |

`NgramTableRows` (`qwen4_exp_weights.cpp`) already took the stated set as the
authority, and `Qwen4ExpPleParams::head_vocab_sizes`'s own field comment already
said so in those words: "Where the source states them they are the AUTHORITY,
because they are what the shipped tensor was actually built against."
`Qwen4ExpPleLayout` did not. It built `head_vocab_sizes`, `head_offsets` and
`padded_vocab_size` from the prime chain, ALWAYS, and then refused when a stated
set disagreed with the chain. On the GGUF arm the chain's input is a defaulted
20,000,000, so:

1. **The comparison was against a default, never against the file.** It can hold
   for exactly one artifact in existence — the released checkpoint, whose base
   really is 20,000,000 — and refuses every other `qwen4exp` file with correctly
   loaded weights. That is what stopped W5f inside layer 1.
2. **Had it not refused, this was a HEAP OVER-READ and not merely wrong rows.**
   This is the W5g review's correction to an earlier wording here, which said
   the layout "would have been wrong" and left the reader to infer how badly.
   The mechanism is exact, and it is worth stating because the guard that looks
   like it would catch this is the thing that fails:

   `qwen4_exp_ple.cpp:293` picks the gather row as
   `(mixed % layout.head_vocab_sizes[head]) + layout.head_offsets[head]`, and
   `:389` bounds-checks it with
   `if (row < 0 || row >= layout.padded_vocab_size)` — **against the LAYOUT's own
   `padded_vocab_size`, not against the rows the buffer actually holds.** So when
   the layout and the tensor disagree, the check is computed from the same wrong
   number that produced the row, and it passes.

   The fixture makes the size of the gap concrete. It states head vocabularies
   23 and 29; `NgramTableRows` returns the stated padded size, so
   `per_layer_token_embd.weight` is allocated with **128 rows**. Pre-W5g
   `Qwen4ExpPleLayout` derived the chain from the defaulted base 20,000,000,
   giving head sizes 20,000,003 and 20,000,023, offsets 0 and 20,000,003, and
   `padded_vocab_size = 40,000,128`. Row indices therefore ranged over
   `[0, 40,000,026)` and the bounds check admitted every one of them, while
   `weights.ngram_embedding + row * head_dim` addressed a 128-row allocation.
   Essentially every gathered row would have read past the end of the buffer —
   an out-of-bounds read of up to ~40 M rows beyond it, with no shape error and
   no refusal anywhere on the path.

   The refusal in consequence 1 was the only thing standing between the GGUF arm
   and that read, which is why "narrow the cross-check" on its own — W5e-2's
   option (b), and one of the two repairs W5f proposed — would have converted a
   loud refusal into a silent over-read. The released checkpoint escapes it
   because its own base IS 20,000,000, so the chain reproduces the file; that
   coincidence, not the code, is why nothing observed this until a forward ran
   the PLE layer on a small fixture.

So the repair is neither of the two W5e-2 listed. **(a) teaching the GGUF config
builder an optional `qwen4exp.ple.ngram_vocab_size_base` is refused**: it invents
a container key the container oracle (`llama-cpp-qwen4exp`, ggml-org/llama.cpp
PR #27742) does not write, so the fixture would exercise our invention rather
than the container, and consequence 2 above would survive untouched for every
real file. **(b) narrowing the cross-check is right and insufficient on its own**
— narrowing silences the refusal and leaves the wrong offsets. W5g does (b) AND
promotes the stated set to the authority for the layout, which is the rule the
rest of the port already followed.

### What W5g changes

- `Qwen4ExpPleParams` gains `head_offsets` (the container's own array, written
  into the text config by `Qwen4ExpHfConfigFromGguf` since W6a and read by
  NOTHING until now) and `ngram_vocab_size_base_stated` (whether the SOURCE said
  it; the VALUE cannot say, because 20,000,000 is both upstream's default and the
  released checkpoint's own base).
- `ParseQwen4ExpParams` checks a stated offset array against the stated sizes.
  The defect that catches is silent: the offsets select rows inside a table whose
  row count both arrays agree on, so a wrong one gathers another head's vectors
  and no shape anywhere is wrong.
- `Qwen4ExpPleLayout` builds the layout from the stated set when there is one,
  derives the chain when there is not, and runs the chain-vs-stated cross-check
  only where the SOURCE stated the base. `layer_multipliers` stays the
  derivation's: it is a splitmix chain over `vocab_size`, `ngram_size`, the layer
  index and `seed`, none of which the stated arrays carry.
- A stated set on a PLE layer OTHER than index 0 is REFUSED BY NAME. Upstream
  derives a different vocabulary per PLE layer from the global head index
  (`ple_layer_index * ngram_heads + head`), while the container states one flat
  array of `ngram_heads` entries and says nothing about which layer it describes.
  The released checkpoint has one PLE layer, so index 0 is the only unambiguous
  case. Silence here would give layer 1 layer 0's vocabulary.

Nothing is widened and no tolerance moved. The old refusal keeps its message and
its three subcases and now fires exactly where it can be true.

### Mutation evidence

Each guard was deleted in place, rebuilt, run, and the tree restored (`git diff`
empty, `touch` after restore so ninja could not skip).

| mutation | what was removed | result |
|---|---|---|
| MUT-REACH | the `Qwen4ExpTextModelForward` call site in `qwen4_exp_registry.cpp`, replaced by a zeroed `[T, hidden]` | `test_qwen4_exp_layer_loop` RED, 90/91, at `CHECK(hi > lo)` |
| MUT-AUTHORITY | `layout.head_vocab_sizes.assign(stated...)` — keep the derived chain | `test_qwen4_exp_layer_loop` RED at `ple_block.cpp:218`, `head 1 offset disagrees ... 23 ... 20000003` |
| MUT-NARROW | the `ngram_vocab_size_base_stated` condition — compare always | `test_qwen4_exp_ple_block` RED **and** `test_qwen4_exp_layer_loop` RED with W5f's exact blocker restored |
| MUT-OFFSET | the container-offset cross-check in `Qwen4ExpPleLayout` | `test_qwen4_exp_ple_block` RED, `did NOT throw at all` |
| MUT-LAYERIDX | the `ple_layer_index == 0` guard | `test_qwen4_exp_ple_block` RED, `did NOT throw at all` |
| MUT-PARSE | the whole `ple_head_offsets` validation in `ParseQwen4ExpParams` | `test_qwen4_exp_scaffold` RED on all three subcases |
| MUT-INPUT | (W5g review repair) the hook IGNORES `token_ids`: `ForwardQwen4ExpForConditionalGeneration` forwards a fixed `std::vector<int32_t>(T, 1)` in its place, every other input untouched | build rc 0. `test_qwen4_exp_layer_loop` RED **103/104, on `CHECK(moved > 0.0)` alone**, `moved = 0`. The golden case stays GREEN at `0.00982457` |

**MUT-REACH is the finding worth reading twice.** With the whole 4-layer tower
deleted, the hook still returned `[1, 16]` logits, every element finite, and
`vt::GreedyArgmax` still sampled a token — id 0, from an all-zero row. Shape,
finiteness, range and "the argmax is the row's maximum" ALL passed. Only
`CHECK(hi > lo)` fired. A reachability case that asserted "a token came out"
without asserting that the logits VARY is a mute switch, and this row already
found that exact defect in another form today.

**AND `hi > lo` IS NOT THE WHOLE REPAIR, which is the W5g review's finding.**
Because MUT-REACH reds on exactly ONE assertion, that one assertion is the whole
of what separates a reached tower from an unreached one — and all it says is that
the row is not CONSTANT. `lm_head` times ANY non-constant hidden clears it. So a
hook that never read `token_ids` at all would have passed the case as written:
90 assertions that do not look at the prompt, plus one that only asks the output
to vary. "A token came out" was not yet "a token came out of THIS prompt".

The case now runs a SECOND prompt on FRESH caches and asserts the logits moved
(`vllm_test::MaxAbsDiff(host, host2) > 0.0`, measured at **0.0546875**). The
caches are rebuilt rather than reused because the first forward writes the paged
KV and both recurrent states, and a difference sourced from a dirty cache would
let a `token_ids`-ignoring hook pass this assertion too. This is the sibling
row's own shape one file away — `test_glm5_next_forward.cpp` runs a second `Step`
and asserts the same property — and MUT-INPUT above is the proof that it bites:
with the prompt ignored, `moved` reads exactly 0 and 103 of the case's 104
assertions still pass.

### `multi_kv` is NOT lifted, and W5f is not the condition that would change that

The prior investigation said the guard must not be lifted without a consuming
forward. W5f is a forward and it is NOT a consuming one, so the premise did not
in fact move. `ForwardQwen4ExpForConditionalGeneration` reads `input.attn_kv` and
`input.gdn_state` — the POSITIONAL channels — and never touches
`input.multi_kv`. Three independent reasons lifting would be wrong today, in
increasing order of how quickly they bite:

1. **It would not produce a token, it would produce a different refusal.** The
   runner publishes three groups for `qwen4_exp`, of which TWO are
   `AttentionSpec` (group 0 QSA paged, group 2 the MLA indexer side cache), so
   `attn_kv` arrives carrying both groups' caches while the hook checks
   `attn_kv.size() == n_qsa` (`qwen4_exp_registry.cpp:254`). The step would stop
   there instead.
2. **The second step has nowhere to live.** The hook refuses `past_len != 0`
   (`qwen4_exp_registry.cpp:221`) because `ModelForwardInput` carries no home for
   the QSA indexer side cache or the PLE conv ring and n-gram history. A lifted
   guard buys step 1 and refuses at step 2 — which is not decode.
3. **A refusal nothing can drive red is a claim, not a guarantee.** Letting a
   shape through that nothing consumes is exactly the mute-switch the guard
   exists to prevent, and `CHECK_THROWS` satisfied by an unrelated exception is
   the defect this row already found once today.

**What WOULD change the answer**, precisely: a forward that resolves its caches
through `MultiKvCacheIndex::Resolve` by layer name — which `ENG-MULTIKV-BYNAME`
now makes possible for recurrent members too, 5 of 5 on this shape where it was
2 of 5 — together with a `ModelForwardInput` that can carry the two states no
channel carries. Then the guard becomes the per-architecture capability bit
`.agents/specs/kv-dsv4-multicache.md` describes, and this row owns its arm.

**SETTLED BY W5j, AND THE THREE REASONS ARE WHY IT TOOK A WAVE OF ITS OWN.** This
section stays as written because it is the argument that scoped W5j, and each
numbered reason is answered rather than withdrawn. (1) held exactly: `attn_kv`
does arrive at 2 x n_qsa and the assertion this section names as
`qwen4_exp_registry.cpp:254` would have produced a different refusal, so W5j
resolves BY NAME and leaves that assertion guarding the positional arm alone.
(3) held: the capability bit landed WITH its consumer and M4 drives the refusal
red by clearing it, so it is not a claim. (2) is the ONE that did not fully move
— the hook still refuses `past_len != 0`, but its REASON changed: the indexer
side cache now persists in the engine's group-2 pages and the PLE conv ring and
n-gram history are published in the recurrent group, so what blocks the second
step is a dtype and a residency the block refuses, not a missing home. See
`## Mutation record — W5j` and the PLE entry under `## Owed`.

## Mutation record — W5h (#2031, issue OWED)

**THE INDEXER SIDE CACHE WAS FOUR TIMES TOO SMALL, AND THE VIEW OVER IT WAS
FOUR TIMES TOO LARGE.** Found by taking the serving question seriously: the
runner allocates every published cache on the multi-cache path, and this is the
one the model cannot read.

**WHAT W5c-1 PUBLISHED.** Group 2 as
`MLAAttentionSpec(..., compress_ratio = indexer_compress_ratio)`, i.e. 4. That
makes `storage_block_size()` = `block_size / 4` and `page_size_bytes()` =
`storage_block_size() * num_kv_heads * head_size * sizeof(dtype)`
(`src/vllm/v1/kv_cache_interface.cpp:151-152`) — 1024 B per page, 64 B per token
per layer, one stored row per FOUR tokens.

**WHAT THE ORACLE SAYS.** transformers 5.16.0
`models/qwen4_exp/modeling_qwen4_exp.py`, the lane pin in
`.agents/oracles/transformers.md` (sha256
`77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`), re-derived
by READING the file rather than relayed:

| Anchor | What it says |
|---|---|
| `modeling_qwen4_exp.py:645-646` | `q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)` — ONE un-normed, un-roped key per TOKEN |
| `:654-655` | `raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)` |
| `cache_utils.py:340`, `:346` | "Update the indexer key cache **by concatenation**", returning `[batch_size, total_len, index_head_dim]` |
| `cache_utils.py:666`, `:672` | the static-cache arm: `[batch_size, max_cache_len, index_head_dim]` |
| `modeling_qwen4_exp.py:678-681` | the POOLED block keys are rebuilt from the raw keys on EVERY step (`index_select` then `.float().mean(dim=1)`); nothing caches a pooled key |
| `:622` | `self.block_topk = self.token_budget // self.compress_ratio` — where `compress_ratio` actually lives: the indexer's ALGORITHM |

Our own consumer already agreed and nobody had compared the two:
`Qwen4ExpQsaPagedCaches::index_key` is `[max_kv, indexer_head_dim]`, written at
rows `[past_len, past_len + T)` (`qwen4_exp_qsa_block.cpp:426`) and read over
rows `[0, complete_keys)` (`:193`). One row per token, both sides.

**THE SECOND CONSEQUENCE IS AN OVERRUN, NOT MERELY A SHORT CACHE.** The runner
allocates `num_blocks * page_size_bytes()` — off `storage_block_size()` — while
the `PagedKvCache` VIEW it hands the forward carries `kv.block_size =
fa_dims[i].block_size`, the spec's own `block_size`
(`src/vllm/v1/worker/gpu/runner.cpp:1532`, filled at `:1333`). At ratio 4 the
view claimed 16 rows per page over an allocation of 4. Unreachable today only
because `ModelRegistry::Forward` refuses a multi-cache topology first, which is
the sense in which the engine guard has been holding a real defect shut.

**THE FIX.** `compress_ratio = 1` for group 2. `storage_block_size() ==
block_size`, allocation and view agree, 256 B per token per layer.

**THE RETIRED REFUSAL.** `block_size % indexer_compress_ratio == 0` guarded the
integer truncation in `storage_block_size()`. At ratio 1 there is no division.
It is deleted rather than weakened, and its replacement is strictly stronger: a
CAPACITY LAW asserting the side cache's row count equals the paged K/V group's
token count, evaluated at a dividing block size (16) AND at the one the old
refusal rejected (18). The old refusal could see one arithmetic accident; the
law sees any spec that cannot hold what the model stores. The retired SUBCASE
is kept as an ACCEPTANCE (`block_size` 18 must NOT throw), so reinstating the
ratio reds in two places rather than silently restoring a passing test.

**THE LAW IS DERIVED, NOT A CONSTANT.** `token_slots` comes from group 0's own
`block_size * num_blocks`, never from the literal 16. M3 below is the paired
red.

### Gate

Base `e2d58307162b7505b5f6a3039b0e6688954dc90b` (`row/MODEL-MM-QWEN4-EXP-E2E`).
CPU-only build, `cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
-DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF`, `-j 2`, named targets.

| Suite | Before | After |
|---|---|---|
| `test_qwen4_exp_kv_cache` | 4 cases / 399 assertions, `SUCCESS` | 5 cases / 414 assertions, `SUCCESS` |
| `test_qwen4_exp_layer_loop` | 2 cases / 104 assertions, `SUCCESS` | 2 cases / 104 assertions, `SUCCESS` |

The transformers 5.16.0 end-to-end golden is UNMOVED: `max|diff| = 0.00982457`
against a bound of `0.03`, on both sides of the change.

**RED FIRST, and the numbers are the defect.** The new case was committed
against the UNCHANGED product and read:

```
CHECK( indexer_rows == token_slots ) is NOT correct!
  values: CHECK( 32 == 128 )
  logged: the indexer side cache holds 32 rows for 128 token slots
CHECK( idx->storage_block_size() == idx->block_size ) ... CHECK( 4 == 16 )
CHECK( idx->compress_ratio == 1 ) ................. CHECK( 4 == 1 )
CHECK( idx->page_size_bytes() == ... ) ............ CHECK( 1024 == 4096 )
CHECK( idx->page_size_bytes() / idx->block_size == 256 ) ... CHECK( 64 == 256 )
CHECK( token_slots <= indexer_rows ) ............. CHECK( 128 <= 32 )
test case THREW exception: qwen4_exp KV spec: block_size 18 is not a multiple
  of `indexer_compress_ratio` 4 ... at qwen4_exp_registry.cpp:516
```

### Mutations

Build rc read BEFORE any test output in every case; a build failure is not a
test result. Applied-proof is the post-mutation sha256 against the pre-mutation
one, restore-proof is the sha256 back to
`439d2998a704d3b745cd6768980a4159a267ded3df2a36b1ff1550abbca8b6bb`
(`qwen4_exp_registry.cpp`) and
`865c3b7fd11c4ae6619958226749819cc265bee551f27fd7e476364e5325c0f7`
(`test_qwen4_exp_kv_cache.cpp`). `touch` after every restore, or ninja skips the
rebuild.

| # | Mutation | sha256 after apply | Build rc | Result |
|---|---|---|---|---|
| M1 | `compress_ratio` 1 -> `p.qsa.compress_ratio` (the pre-W5h value) | `693f6d8bb868c2b71749f4eddab261a0e8f84becce92c04c9ecf8e08c50c486c` | 0 | **RED** — 3 of 5 cases, 13 of 414 assertions |
| M2 | REACHABILITY: delete the group-2 `emplace_back` from `MakeQwen4ExpKVCache` | `6d555e75006cb8f4be49fbf56012d3eb1c4cf629ed69fc731618bbdd0eaf6dfa` | 0 | **RED** — rc 139, 3 failed cases, `SIGSEGV` in the pre-existing group-2 case. NOT vacuous: the suite reaches the production `make_kv_cache` through `ModelRegistry::Resolve(config).factory`, which is the entry `LoadedEngine::MakeKVCacheMaybeSpec` calls |
| M3 | group 0's `FullAttentionSpec` built at `block_size * 2` | `6cb6b8bb56848e94b3588860d3574cc5afce5df86e5054d3217c938cb880d3bf` | 0 | **RED** — 3 assertions, and exactly the three derived ones (`indexer_rows == token_slots`, `token_slots <= indexer_rows`, the block-18 law). The shape assertions stayed green, which is what proves the law reads group 0 and not the literal 16 |

**THREE MUTATIONS THE DISPATCH ASKED FOR ARE VACUOUS ON THIS WAVE AND ARE
REPORTED AS VACUOUS RATHER THAN RUN.** "Resolve a cache to the wrong name",
"drop the indexer side cache from the resolution" and "lift the guard for a
shape nothing consumes" all mutate code this wave does NOT land: no forward here
resolves by name and the engine guard is untouched. A mutation whose target does
not exist cannot red for the reason claimed, and running one anyway would be the
"a mutation that never applied reads as a passing test" failure. They belong to
W5i and W5j below.

## Mutation record — W5i (#2031, #2249 item 3, issue OWED)

**W5i makes the QSA INDEXER SIDE CACHE readable and writable through the PAGED
allocation.** `Qwen4ExpQsaPagedCaches::index_key` becomes the runner's own fused
MLA page `[num_pages, block_size, indexer_head_dim]` and gains
`index_block_table`, GROUP 2'S OWN logical-to-physical page map. The store
scatters with `vt::IndexCopy` and the read gathers with `vt::IndexSelect`, both
at physical rows resolved by `IndexerRows`.

**NO NEW OP AND NO OP EXTENSION, and the arm count is why.** W5h's `## Owed`
claimed `vt::IndexSelect` and `vt::IndexCopy` suffice; that was re-verified
against their contracts before any code was written and it holds —
`IndexCopy`'s destination must be a contiguous `[N, D...]`, which is exactly
what an `MLAAttentionSpec` group's pages are when flattened, and both ops
already register a `kCPU` AND a `kCUDA` arm. The alternative was W5d-3's
precedent, an ADDRESS MODE on the op: `vt::Qwen4ExpQsaCompress` is the op that
reads this cache and it has **ONE arm** (`kCPU`, `src/vt/cpu/cpu_qwen4_exp_qsa.cpp`;
there is no CUDA arm for any `qwen4_exp` op). That count is what decided it in
the OPPOSITE direction from `vt::RmsNormGroup`, which became its own OpId
because `kRmsNorm` has six arms and a silently-ignored field would return a
wrong answer on five of them. Here a one-arm op with no gate-able second arm
would take an address mode nothing could measure, so the composition is the
smaller and the more honest change.

### Gate

Base `e12a197cd71a596b7d520264bed0804685c5c222` (`row/MODEL-MM-QWEN4-EXP-SERVE`).
CPU-only build, `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF`, `-j 2`,
named targets only. Every `Before` figure was MEASURED on a scratch worktree at
that base SHA and not carried over from an earlier record.

| Suite | Before | After |
|---|---|---|
| `test_qwen4_exp_qsa_block` | 11 cases / 4382 assertions, `SUCCESS` | 12 cases / 5937 assertions, `SUCCESS` |
| `test_qwen4_exp_layer_loop` | 2 cases / 104 assertions, `SUCCESS` | 2 cases / 104 assertions, `SUCCESS` |
| `test_qwen4_exp_kv_cache` | 5 cases / 414 assertions, `SUCCESS` | 5 cases / 414 assertions, `SUCCESS` |
| `test_qwen4_exp_forward` | 1 case / 421 assertions, `SUCCESS` | 1 case / 421 assertions, `SUCCESS` |

The transformers 5.16.0 end-to-end golden is UNMOVED: `max|diff| = 0.00982457`
against a bound of `0.03`, and `ModelRegistry::Forward` still samples token 15.

**RED FIRST, AND THE RED IS THE FINDING.** The case was committed against a
STRAWMAN body that flattened the paged pages and addressed them LINEARLY — which
is mutation M3, so the red-first evidence and M3's proof are the same
measurement made twice. It read:

```
CHECK( wrong_written == 0 ) is NOT correct!
  values: CHECK( 9 == 0 )
  logged: paged-indexer block max relative difference vs the oracle 0.00558036
          paged vs contiguous differing bf16 words 0 of 1472
          named rows left unwritten 9; unnamed rows written 9
CHECK( wrong_untouched == 0 ) ...................... CHECK( 9 == 0 )
CHECK( std::isnan(... tail_row ...) ) .............. CHECK( false )
CHECK( rel < kOutTol ) ............... CHECK( 1.3819 <  0.03 )
CHECK( differing == 0 ) .............. CHECK( 64 == 0 )
CHECK( std::isfinite(... row22 ...) ) CHECK( false )
```

**READ THE THIRD AND FOURTH LINES OF THAT BLOCK BEFORE WRITING ANOTHER GATE ON
THIS AXIS.** With 9 of 23 rows in the wrong physical page, the paged-vs-
contiguous comparison read `differing 0 of 1472` and the oracle bound read
`0.00558036` — both GREEN. The store and the read share one translation, so a
translation that is wrong THE SAME WAY on both sides writes and reads the same
wrong rows and returns the right answer. A value comparison at prefill is not a
gate on paging. What convicts is the STRUCTURAL assertion (the exact set of
physical rows written, against the set the block table names, every other row
still the NaN it was constructed with) and a decode step whose prefix the TEST
places at rows it computes itself.

The fixture is `{2, 6, 1}` for the indexer against `{5, 3, 7}` for the K/V —
TWO DIFFERENT permutations, because group 0 and group 2 are separate physical
page pools and one table could not see a body that resolved the indexer through
the K/V map. Neither shares a fixed point with the logical `{0, 1, 2}`. 23
tokens over pages of 8 leave the third page PARTIAL, so row 7 of it is the one
row a full-page defect reaches for and it is asserted NaN by name.

### Mutations

Build rc read BEFORE any test output in every case; a build failure is not a
test result. Every mutation below was run against the FINAL tree, after the
record and comment repairs and after the three refusal subcases were added —
the first battery ran against an earlier tree and is not reported, because a
later commit silently disarms an earlier commit's mutation proof. Applied-proof
is the post-mutation sha256; restore-proof is the sha256 back to
`bb9baf661a248b74eb26281a3f79d1115871c6a16392e65f83e7fc2f503a844e`
(`qwen4_exp_qsa_block.cpp`) and
`6e3d8b64612f28d90da37ce08bf28cd145ab4f60411350d559e4c49510c21a21`
(`qwen4_exp_registry.cpp`), verified by `diff` after each. `touch` after every
restore, or ninja skips the rebuild.

| # | Mutation | sha256 after apply | Build rc | Result |
|---|---|---|---|---|
| M1 | OFF-BY-ONE PAGE INDEX in `IndexerRows`: `lp = (pos + 1) / block_size` | `a491df61e800c7527847f190b4028fdae8c6e6d5c84bf1a6f726601a820eb853` | 0 | **RED** — 1 of 12 cases, 3 of 5937 assertions, all three the STRUCTURAL ones (`wrong_written 1`, `wrong_untouched 1`, the tail row). The value subcase stayed GREEN, which is the shared-translation property above measured a second time |
| M2 | PARTIAL FINAL PAGE READ AT FULL LENGTH: the gather rounds the visible prefix up to a whole page and tells `Qwen4ExpQsaIndex` so | `da0349e30dacb1746f3ab1ee1cc64cefe9fdc3f4cab09ba156500e46822ed3a2` | 0 | **RED** — 3 of 12 cases (all three paged cases), by a THROW rather than an assertion: `qwen4_exp_qsa_compress: cos/sin must cover every key position`. Reported with the mechanism because it is not the one intended — the NaN tail row is UNREACHABLE without overstating the prefix, and overstating it is caught first by a guard that predates this wave. `assertions: 2844`, all passed, is a thrown case and not a green one |
| M3 | DROP THE PERMUTATION: `IndexerRows` returns `pos`, ignoring the table | `60b4c20e08434bea4123179381549c8f4eae0a77d90d71dd0854e29fcdd3e03a` | 0 | **RED** — 1 case, 6 of 5937 assertions, across BOTH subcases: the three structural ones and the decode's `rel 1.3819 < 0.03`, `differing 64 of 64`, `isfinite(row22) false` |
| M4a | REACHABILITY, fatal-body form: `IndexerRows` opens with `VT_CHECK(false, ...)` | `cd88b512622a0946f029659e5b9dbf25789bcb31195b576e067e3e4752facb26` | 0 | **RED** in `test_qwen4_exp_layer_loop`, 2 of 2 cases, including `REQUIRE_NOTHROW( fl = vllm::ModelRegistry::Forward(*model, in) ) THREW`. **NOT VACUOUS**: the paged indexer translation is reached from `ModelRegistry::Forward` on a loaded `qwen4exp` GGUF, which is a production entry point |
| M4b | REACHABILITY, call-site-deletion form (`.agents/reachability.md` step 5): the registry hook's paged scratch goes back to the pre-W5i contiguous `[T, D]` and the `index_block_table` assignment is deleted | `cd726d05fa565d06820eb81cdc13e15348e9f4f4d1068319bd9b187f9d0f4ed8` (`qwen4_exp_registry.cpp`) | 0 | **RED** on the `ModelRegistry::Forward` case alone, with the block's own refusal: `the paged indexer side cache must be a contiguous [num_pages, block_size, indexer_head_dim]`. The oracle case stayed GREEN, which is the separation the step exists to produce — it drives `Qwen4ExpTextModelForward` with its own caches and does not go through the hook |

**WHAT NO MUTATION HERE PROVES.** M4a and M4b prove the translation is reached;
neither proves the ENGINE's group-2 buffer reaches it, because it does not — the
`multi_kv` refusal stands and the hook substitutes a scratch. The scratch's
block table is the IDENTITY, so the PERMUTATION is exercised only by the block's
gate and never on the production path. That is stated in `## Owed` and it is
W5j's to close.

## Mutation record — W5j (#2031, #2353, issue OWED)

**W5j makes the forward CONSUME the by-name channel and narrows the engine's
`multi_kv` refusal to a model-declared capability.** Three things stood between a
`GPUModelRunner` step and this hook, and all three are gone:

1. **The hook read the caches POSITIONALLY.** It now resolves every published
   cache through `MultiKvCacheIndex::Resolve` when the channel is present —
   group 0's paged K/V under `model.layers.<N>.self_attn.attn`, group 2's indexer
   side cache under `model.layers.<N>.self_attn.indexer.k_cache`, and each
   recurrent state under `model.layers.<N>.linear_attn`, which is the recurrent
   arm `ENG-MULTIKV-BYNAME` added and which #2353's survey recorded as
   unaddressable.
2. **`attn_kv` ARRIVES AT 2 x n_qsa.** The runner allocates one paged buffer per
   (attention group x layer) and this model publishes TWO attention groups over
   the same QSA layers, so the hook's `attn_kv.size() == n_qsa` assertion would
   have turned the lifted guard into a DIFFERENT refusal rather than a token.
   That assertion now guards the POSITIONAL arm only, where it is still exact.
3. **W5i's production path used a per-call scratch behind an IDENTITY table.**
   The by-name arm hands `RunQwen4ExpQsaBlockPaged` the engine's own group-2
   buffer, viewed as the fused MLA page `[num_blocks, block_size,
   indexer_head_dim]`, and group 2's own gathered table from
   `BlockTableForGroup`. W5i said the substitution was "two lines in the hook and
   nothing in the block"; that held — no line of `qwen4_exp_qsa_block.cpp`
   changed except one stale comment repaired in flow (see below).

**THE GUARD DID NOT GO AWAY, IT NARROWED.** `ModelFactory::consumes_multi_kv` is
a model-declared capability bit in the `stage_on_load` / `supports_weight_offload`
style, and it landed WITH its first consumer, which is the condition #2353 named:
"a capability nothing can turn on has no arm a test could drive".
`ModelRegistry::Forward` still refuses any multi-cache topology reaching a
forward that leaves it false, which is `DeepseekV4ForCausalLM`,
`Glm5NextForConditionalGeneration` and every model ported after them. It is NOT a
licence to serve any shape: the hook refuses a name nothing was published under,
a name resolving to the wrong payload kind, and a group whose block table was
never gathered, each by name at its own boundary.

### What this does NOT do: a second step still does not decode

`past_len > 0` is still refused, and W5j CHANGES ITS REASON rather than removing
it. The old message said the QSA indexer side cache and the PLE conv ring and
n-gram history had "nowhere to persist"; two thirds of that is now false — the
channel reaches the hook and group 2's pages persist. What remains is the PLE
layer's PAIR OF STATES, and it is a DTYPE and a RESIDENCY rather than a missing
channel:

- `MakeQwen4ExpKVCache` publishes the PLE conv ring at the recurrent group's
  uniform `conv_dtype`, which is bf16, and `RunQwen4ExpPleBlock` requires f32
  (`qwen4_exp_ple_block.cpp:320`) because it walks the ring through `Ptr<float>()`.
- It publishes the n-gram history as a DEVICE i64 state, and the same block
  requires it HOST-resident (`:327`, read through a host pointer at `:351`)
  because the splitmix64 hash is a host int64 computation.

Which side of each is wrong is an ORACLE question. W5h called it "a design call I
did not have the oracle to settle" and it is not settled here either: the lane
pin's `modeling_qwen4_exp.py` is not readable on this host — the local
`transformers` checkout is at `7d06b1a5` and has no `models/qwen4_exp/` directory,
and the installed wheel is 5.3.0. Per the row's own rule this is RECORDED under
`## Owed` and not guessed at. **Nothing decoded a second token in this wave, on
any device.**

### Gate

Base `7c8b06b87de2bbc41932abf773405bf781e88719` (`row/MODEL-MM-QWEN4-EXP-W5I`).
CPU-only build, `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
-DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF`, `-j 2`, named targets only.

| Suite | Before | After |
|---|---|---|
| `test_qwen4_exp_layer_loop` | 2 cases / 104 assertions, `SUCCESS` | 4 cases / 197 assertions, `SUCCESS` |
| `test_qwen4_exp_kv_cache` | 5 cases / 414 assertions, `SUCCESS` | 5 cases / 414 assertions, `SUCCESS` |
| `test_qwen4_exp_scaffold` | not measured separately | 12 cases / 305 assertions, `SUCCESS` |
| `test_qwen4_exp_qsa_block` | not measured separately | 12 cases / 5937 assertions, `SUCCESS` |
| `test_qwen4_exp_ple_block` | not measured separately | 12 cases / 100 assertions, `SUCCESS` |
| `test_model_registry` | not measured separately | 24 cases / 975 assertions, `SUCCESS` |
| `test_registry_downcast_refusal` | not measured separately | 6 cases / 33 assertions, `SUCCESS` |
| `test_runner` | not measured separately | 36 cases / 1851 assertions, `SUCCESS` |
| `test_glm5_next_forward` | not measured separately | 9 cases / 118 assertions, `SUCCESS` |
| `test_glm5_next_scaffold` | not measured separately | 38 cases / 2652 assertions, `SUCCESS` |

The `test_qwen4_exp_layer_loop` `Before` row is MEASURED on this worktree at the
head that carried the hook and guard changes and NOT the new test cases, which is
what makes it a statement about the POSITIONAL arm: it is byte-identically the
count W5i recorded. The seven suites marked "not measured separately" were run at
the W5j head only; each is green and none of them is claimed as a delta.
`test_qwen4_exp_qsa_block` is the one that matters most among them: W5j changed
no line of `qwen4_exp_qsa_block.cpp` except one stale comment, and its 5937
assertions are the evidence for that claim rather than a reading of the diff.

The transformers 5.16.0 end-to-end golden is UNMOVED: `max|diff| = 0.00982457`
against a bound of `0.03`, the same value W5f, W5g, W5h and W5i recorded.

`scripts/agent-preflight.sh --fail-on-skip` exits 1 with ONE failing gate,
`role-undeclared`, which is the operator's record and not this change's. The
first run, against `origin/main` at `7d53ae3b4`, reported **zero SKIPs** and
`ok commit-trailers` / `ok commit-style`. A second run minutes later reported
TWO SKIPs on exactly those two gates, because `origin/main` had moved to
`03e0dcd19` underneath a shared checkout and the checker refuses to judge a
branch its base is not an ancestor of. That is structural for this whole local
stack — the base is `7c8b06b87`, which is not on `main` — and not something this
wave introduced, so the two gates were run DIRECTLY against this wave's own
range instead: `check-commit-trailers.py --range 7c8b06b87..HEAD --filled` and
`check-commit-style.py --range 7c8b06b87..HEAD` both exit 0. Recorded rather
than reported as a clean sweep, because a SKIP is not an `ok`.

### Mutations

Every mutation was sha256-proved applied before the build, the build rc was read
BEFORE any test output, and the tree was restored with a sha256 equality check
against the pre-mutation digest. Base digest of
`src/vllm/model_executor/models/qwen4_exp_registry.cpp`:
`f4f60b86d80b1bc4e64a3e39d621ae52c5bd61d9caf0cca5040ccf09f5e958b8`. All six were
RE-RUN against that digest after the shared-name-builder refactor below, so the
record describes the bytes that ship and not an earlier head.

| ID | Mutation | Build rc | Result |
|---|---|---|---|
| M1 | resolve the side cache to a WRONG NAME (`Qwen4ExpQsaIndexerName(l)` replaced by a literal `...self_attn.indexer.kcache`) | 0 | RED, `4 cases / 2 passed / 2 failed`, `145 assertions / 5 failed`. The by-name case's `REQUIRE_NOTHROW` threw the hook's own "the engine published no KV cache under 'model.layers.3.self_attn.indexer.kcache'", and the wrong-name SUBCASE's name assertion flipped. A first attempt appended `"X"` to the shared builder's result instead and separated by only 4 assertions, because `...k_cacheX` still CONTAINS the substring the SUBCASE looks for — recorded because it is the same substring-containment trap this row's two-sided message assertions exist to avoid |
| M2 | read the side cache through GROUP 0's block table (`group_ids[k.first]` to `group_ids[a.first]`) | 0 | RED, `197 assertions / 7 failed`, on `got_a == expect_a`, `got_a != via_group0`, `got_b != got_a` and the third run's row set. **NOT ONE VALUE ASSERTION MOVED** — the logits, the two-map drift and the second-prompt movement all stayed green, which reproduces W5i's measured lesson inside this wave's own gate |
| M3 | drop the engine's group-2 buffer and keep the per-call scratch | 0 | RED, `145 assertions / 2 failed`. The step threw `vt: index_copy: idx out of range` because a permuted table names a page a T-sized scratch does not have. Detected, but by the OP and not by an assertion, so the row-set gate never ran — recorded as the weaker of the six |
| M4 | clear `consumes_multi_kv` — the guard must still refuse | 0 | RED, `145 assertions / 12 failed`. `ModelRegistry::Forward` threw its own message naming the architecture, `5 KV cache(s) (2 paged, 3 recurrent) from 3 published group(s)` and "its ModelFactory leaves `consumes_multi_kv` false", and every `kEngine`-absent half of the three two-sided refusal assertions flipped. This is the "must still refuse" arm |
| M5 | drop the indexer side cache from the resolution entirely, AND delete the adjacent same-slot guard so no neighbouring check can catch it | 0 | RED, `197 assertions / 10 failed`, on the row-set assertions and the wrong-name SUBCASE. Again NO value assertion moved |
| M6 | REACHABILITY: delete the `Qwen4ExpTextModelForward` production call site, substituting a zeroed hidden of the right shape | 0 | RED, `197 assertions / 7 failed`. NOT VACUOUS: the call site exists and the by-name case depends on it, reddening `hi > lo`, both row-set assertions, the two-map row-set difference and the second-prompt movement |

**What the six do NOT prove.** No mutation deletes `RunQwen4ExpPleBlock`'s call
site, so the PLE ops remain reached-by-argument on this row, exactly as W5i left
them. And M3's red is an op range check rather than an assertion, so the by-name
buffer substitution is proved DETECTED but not proved detected BY THE GATE.

### ONE builder for the published names, which the spec asked for

`## Owed`'s W5j entry said the hook "must build its layer names through ONE
builder shared with `MakeQwen4ExpKVCache` — two derivations of one name set is
the shape that can disagree". A first cut of this wave deliberately did the
OPPOSITE and argued for it in the source: that a shared helper would make a
publisher/consumer disagreement invisible. That argument is wrong in the
direction that matters. A disagreement between the two is a RUN-TIME refusal with
no compile error behind it, and making it impossible by construction is strictly
better than making it detectable; the resolution failure then means only that the
ENGINE did not carry what the model published, which is the one thing the refusal
should be able to say. The three builders are file-local — both sides live in
`qwen4_exp_registry.cpp`, so exporting them would only invite a third derivation.
All six mutations were re-run after this refactor.

### Repaired in flow

- `qwen4_exp_qsa_block.cpp`'s paged precondition said "`MakeQwen4ExpKVCache`
  already refuses a `block_size` the compress ratio does not divide". **W5h
  deleted that refusal**, correctly, and the sentence survived. The check at that
  site is still right and its reason is group 0's own — it keeps a compress block
  of CR tokens inside one page — so the comment now says that instead of citing a
  refusal that no longer exists.
- `model_registry.h`'s `multi_kv` field said the topology is "DeepSeek-V4 and
  nothing else". THREE architectures publish one.
- `ModelRegistry::Forward`'s comment said "no registered forward consumes one"
  and "that is still true of all THREE architectures". One consumes one now.

## Owed

### The CAUSE of the three disagreeing token ids ([#2999](https://github.com/mudler/vllm.cpp/issues/2999))

**Narrowed on 2026-09-06, not closed.** `qwen4_exp` emits three different ids at
indices 4, 6 and 7 between `--device cpu` and `--device cuda`, from one binary,
one set of weights and — since #2612 — one algorithm on both sides. **Nothing
measured on this row explains why those three.**

Wave MOESEL-RESULT
([reading](../../docs/bench-evidence/qwen4exp-moe-selection-fwd-20260906.md), `rc`
job `9e0864da`) removed two candidate explanations from the unknown column and
added none:

- Expert selection at the disagreeing forwards **has now been read**, so the
  standing "no instrument has observed a disagreeing step" clause is discharged.
- The selection **does** flip between the arms, 296 of 576 slots, on a comparator
  whose negative control flips nothing.
- **The flips do not separate the two groups.** Forward 5 flips 48 of 48 slots and
  its id agrees; forward 7 flips 48 of 48 and its id disagrees; forward 4
  disagrees while flipping 33 of 48. A per-forward flip count is therefore not the
  cause, in either direction.

**The next traceable step.** The selection tap answers "did the chosen experts
change". It does not answer "did the change move the logit `argmax` reads".
Separating those needs a tap on the routed output or the final logit row at
forwards 4, 6 and 7, compared against the four agreeing forwards, on the same
matched pair. It needs the 68 GB released UD-IQ1_S artifact and a GPU for the CUDA
arm, so it is a lease.

**Two constraints on whoever takes it.** Do not read a flip count as a cause; that
is the error this entry exists to prevent. Do not publish a `rel(sumabs)` ratio for
any of it, because that axis is a difference of norms and cannot rank magnitudes
(§"needs a DIFFERENCE-NORM axis").

### The CPU-vs-CUDA TOKEN-IDENTITY objective is UNREACHABLE AS WRITTEN (#2831)

### The `VT_Q4EXP_LAYER_FP` tap needs a DIFFERENCE-NORM axis ([#2877](https://github.com/mudler/vllm.cpp/issues/2877))

**Why the differ repair does not close this.** `scripts/q4exp-layerfp-diff.py`
landed with #2877 and fixes the layer-axis collapse, but it can only compare what
the tap emits, and every scalar the tap emits is **sign-insensitive**. `S|x|` is a
norm, so comparing two arms' norms gives a DIFFERENCE OF NORMS: it reads zero on
two tensors that differ in every element but happen to have equal L1 norm, and it
cancels a zero-mean perturbation — which is every reassociation and rounding
difference — at `O(sqrt(n))`. **By a factor that is a DISTRIBUTION, not a
constant**: at this tap's `n = 12800`, over 400 seeds of the hermetic control
committed as `MetricSpread` in `tests/scripts/test_q4exp_layerfp_diff.py`, the
under-report has a median of 75x at sigma 1e-3 and 140x at sigma 1e-4 with a
p05..p95 of 34..1500, and no sigma dependence in the linear regime (1e-4 and 1e-5
agree, both approaching `sqrt(2n/pi)/|z| = 90.3/|z|`, median 134). This spec
first quoted the single draws `122.7x` and `229.8x` to four significant figures,
and a `4.64x` span at a **fixed** true divergence measured over SIX seeds. Over
400 seeds that span is **21x** p05..p95; its end-to-end span is not a figure at
all, because one draw sets it and it moves more than 4x between seed blocks. The
consequence is the one that matters: **two readings of the SAME true divergence
differ by a median 2.1x, 11x at p90 and 24x at p95**, so a ratio in the low tens
between two `rel(sumabs)` numbers ranks nothing in either direction. **Two
significant figures is what 400 draws buy.** The three-figure set this spec
carried until #2879 (`31.4 / 69.2 / 568.2`, `18.3x`, `18.2x`) came from an ad-hoc
script that was never committed and does not reproduce from the control; it is
not a different construction — 14 of its 15 figures lie inside the control's own
bootstrap 95% interval
([study](../../docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904/metric-spread-precision.py))
— it was a different SAMPLE quoted to a digit 400 draws do not determine.
`MetricSpread` now draws every figure the documents quote AND reads those
documents off disk to compare them against it, this spec included
(`test_the_PUBLISHER_*`). The
tool prints `head_dmax`, an exact elementwise difference over the four `v=` values
the tap already emits, which cannot cancel — but it samples 4 of 12800 elements,
so it is a witness and never a magnitude.

**The design.** No per-arm scalar reduction can yield `||a-b||` unless it is
sign-SENSITIVE. Emit a small fixed set of fixed-seed random projections
`p_j = S_i w_ji x_i` beside `sumabs`. Then `p_j(a) - p_j(b) = S_i w_ji (a_i - b_i)`
is a linear functional of the DIFFERENCE itself, and a handful of them estimate
`||a-b||_2` with a known concentration bound, from two independent processes that
never see each other's tensor. The seed must be a compile-time constant so two
arms draw the same `w`, and the projections must be accumulated in `double` for
the same reason `sumabs` is.

**Why it is owed and not done in #2877.** `LayerFp` lives in
`src/vllm/model_executor/models/qwen4_exp_forward.cpp`, on the model forward path.
This is product code, so it needs its own spec, a red-first test, and a fresh
review. #2877 was diagnosis plus record and instrument repair, and it deliberately
changed no `src/` file.

### NO `VT_Q4EXP_LAYER_FP` FINGERPRINT HAS OBSERVED A DISAGREEING STEP ([#2877](https://github.com/mudler/vllm.cpp/issues/2877), [#2969](https://github.com/mudler/vllm.cpp/issues/2969))

`LayerFp` returns early on `s.step >= s.budget`
(`src/vllm/model_executor/models/qwen4_exp_forward.cpp:118`), and **its budget is
counted in FORWARDS**, so every `VT_Q4EXP_LAYER_FP` fingerprint this row has taken
covers model forwards **0, 1 and 2**. Those are the tokens `11751 13 15767`, which
the two arms **agree** on. The three ids that disagree are emitted at forwards
**4, 6 and 7**.
Every ratio published from that tap therefore describes three forwards on which
nothing disagrees.

**What is owed for that tap is a budget that reaches forward 7** on both arms,
which is `VT_Q4EXP_LAYER_FP=8` and needs no code change — the budget is already
read from the environment. It needs the 68 GB released UD-IQ1_S artifact and a GPU
for the CUDA arm, so it is a lease, not a desk run. The `sumabs` axis it would
print is still the one §"needs a DIFFERENCE-NORM axis" above disqualifies for
ranking, so the two are best spent together: the projection axis first, the wider
budget on the same run.

**`VT_MOE_SEL_FP` IS ON A DIFFERENT BUDGET AXIS AND ALREADY REACHES FORWARD 7.**
This is the correction #2969 records, and it changes what one lease buys.
`MoeSelFpCall()` increments once per `MoeBlock` INVOCATION
(`src/vllm/model_executor/models/qwen3_5.cpp:7091`, `:7167`), not once per
forward. `qwen4_exp` runs one MoE block on every one of its 48 layers. The layer
loop at `qwen4_exp_forward.cpp:491` reaches the call at `:691` with no condition,
and `qwen4_exp_weights.cpp:760` loads MoE weights for every layer. So the call
index maps onto the forward index as follows.

| calls | what they are |
|---|---|
| 48 | one model forward |
| 0..47 | forward 0, the `T=5` prefill; the call index IS the decoder layer index |
| `48f .. 48f+47` | forward `f`, a `T=1` decode |
| 192..239 | **forward 4**, the first disagreeing step |
| 336..383 | **forward 7**, the last one |

So wave MOEDIV's 384 calls were **eight forwards**, not a prefill plus an
undifferentiated decode tail, and that tap did reach forwards 4, 6 and 7 on 2
September 2026. Its own committed coverage says so: 384 digests splitting 48
prefill (`T=5`) and 336 decode (`T=1`)
([evidence](../../docs/bench-evidence/qwen4exp-moe-selection-20260902.md)). The
decode columns of that run are VOID for an unrelated and correctly recorded
reason. The CUDA arm answered the degenerate pre-#2550 sequence
`11751 271 271 271 271 271 0 0`, so the comparison ran against a different input.
The instrument did not stop short.

Reaching a disagreeing step with `VT_MOE_SEL_FP` therefore needs no code change
and no budget larger than MOEDIV already spent. What it needs is a non-degenerate
CUDA arm, which #2550 and #2612 have since supplied. **That run was taken on
2026-09-06** ([#2998](https://github.com/mudler/vllm.cpp/issues/2998),
[reading](../../docs/bench-evidence/qwen4exp-moe-selection-fwd-20260906.md)): 384
calls again, `HIGHEST FORWARD REACHED = 7`, and 296 of 576 flipped slots against a
negative control that flips none.

### The ARMTOKENS raw fingerprints survive on the fleet share ([#2877](https://github.com/mudler/vllm.cpp/issues/2877), [#2969](https://github.com/mudler/vllm.cpp/issues/2969))

**This section previously stated that the ARMTOKENS diffs cannot be re-rendered
without a device. That conclusion is withdrawn.** The premise is correct and the
conclusion does not follow from it.

The premise stands: `docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904/`
committed the two rendered diffs and `run2-results.txt`, but **not the raw
`fp.txt` of any arm**. Nothing was mis-committed, because the raw fingerprints are
job output and not repository content.

The conclusion does not stand, because the job output was never lost. All three
ARMTOKENS run-2 arms keep their `fp.txt` on the fleet share at
`/workspace/armtokens-2612/out2/<ARM>/fp.txt`, 1314 lines each. Each file carries
**1311** unique `(step, L, tag)` taps, which is the coverage the committed diff
collapsed to 42. **The collapse is a parser defect and not missing data.** The
differ keys on `f.get('L')`, and the tap prints the layer as `L%+03lld` with no
`=`, so the field never parses and all 48 layers fold onto one key for each
`(step, tag)` pair. Correcting the 42-vs-1311 coverage therefore needs the
repaired differ and the share, and it needs no lease.

Two obligations remain unchanged. A wave that holds a lease on this row must
still capture `fp.txt` for each arm, because the share is fleet scratch and not a
record surface. A reading taken from those files must still respect
§"needs a DIFFERENCE-NORM axis": `rel(sumabs)` cannot rank magnitudes, so a
re-render may state which taps are bit-identical and in what order they stop
being so, and it may not publish a ratio.

### The layer-0 flip question is ANSWERED; the residue is still NOT closed ([#2877](https://github.com/mudler/vllm.cpp/issues/2877), [#2998](https://github.com/mudler/vllm.cpp/issues/2998))

[#2552](https://github.com/mudler/vllm.cpp/issues/2552) bracketed the layer-0
expert-flip threshold between `2.139e-05` (selections agree on all five tokens) and
`4.999e-04` (layer 0 already flips at token 2). The algorithm-matched pair that
#2870 measured — CPU-chunked vs CUDA-chunked — has `L00 mhc.mix` at **`4.324e-05`,
inside that bracket**. **`VT_MOE_SEL_FP` HAS NOW BEEN RUN ON IT**
([evidence](../../docs/bench-evidence/qwen4exp-moe-selection-fwd-20260906.md), `rc`
job `9e0864da`, `thor:gpu0`, 2026-09-06T02:21:41Z, `exit_code = 0`, every stage
`SUM ..._RC=0`).

**The answer is NO FLIP.** All five prefill tokens select the same ten experts on
both arms with an equal `sel` digest, at boundary margins of 4, 2, 1, 1 and 0 ulps,
with the pair identity asserted at call 0 (`x` 3615.47142 against 3615.62777) so a
run on the wrong pair would report as such. **The no-flip bound therefore lifts
from `2.139e-05` to `4.324e-05`**, and the threshold now sits between `4.324e-05`
(no flip) and `4.999e-04` (flip).

**That makes the separation #2877 recorded as unknown, at layer 0.** The matched
pair's layer-0 MoE residue contains no selection flip, so the bimodal top-k term is
not what produces it there; the keep-quant grouped expert GEMM's scale-sum
reassociation is what remains. **Read the scope exactly**: layer 0, this pair, the
prefill. The other 47 layers flip heavily from forward 0 onward — forward 0's first
flip is at layer 4 — so this is not a claim that the top-k term is inert anywhere
else.

**Wave MOESEL ([#2969](https://github.com/mudler/vllm.cpp/issues/2969)) narrowed
what that run costs; wave MOESEL-RESULT read it.** The same budget answered both
open questions at once, because 384 MoE calls is eight forwards and forward 4
begins at call 192, so one lease returned the layer-0 bracket placement and the
selection state at forwards 4, 6 and 7 together. MOESEL qualified the comparator
against committed MOEDIV output first, where it reproduces both the `L00 mhc.mix`
pair 3613.82031 against 3615.62777 and the recorded flip at token 2, and that
qualification is what licenses the reading.

**THE INSTRUMENT HAS NOW OBSERVED THE DISAGREEING FORWARDS.** `HIGHEST FORWARD
REACHED = 7`, on a comparator whose negative control (the same CPU arm run twice)
flips **0** slots on all 8 forwards. The selection flips extensively between the
arms: **296 of 576 slots**.

**AND THE FLIPS DO NOT EXPLAIN THE DISAGREEING IDS.** This is the result's most
important limit, and it is an inversion rather than a weak signal:

| fwd | 0 | 1 | 2 | 3 | **4** | 5 | **6** | **7** |
|---|---|---|---|---|---|---|---|---|
| flipped slots | 53/240 | 22/48 | 29/48 | 17/48 | **33/48** | 48/48 | **46/48** | **48/48** |
| sampled ids agree | yes | yes | yes | yes | **no** | yes | **no** | **no** |

**Forward 5 flips every expert slot in all 48 layers and its id AGREES. Forward 7
flips every slot and its id DISAGREES.** Two forwards with an identical selection
verdict land on opposite sides of the token verdict, and the least-flipping
disagreeing forward (4, at 33/48) flips less than agreeing forward 5. Grouped, the
disagreeing forwards flip 127 of 144 slots and the agreeing ones 116 of 192. **A
flip count separates nothing here, in either direction.** The flip rate rises
roughly with decode depth on both populations, which is consistent with divergence
accumulating in the recurrent and KV state, but that is an observation and not a
mechanism.

**So the residue is still NOT closed and the position is unchanged: the three
disagreeing token ids at indices 4, 6 and 7 remain unexplained by anything measured
on this row.** They are narrowed, not explained. The next traceable step is a tap on
the routed output or the final logit row at the disagreeing forwards on the same
matched pair, because the selection tap reads which experts were chosen and not
whether the choice moved the value `argmax` reads. It needs the 68 GB artifact and a
GPU. Owed under [#2999](https://github.com/mudler/vllm.cpp/issues/2999).

**The position.** The standing objective — `qwen4_exp` emits the CPU control
sequence `11751 13 15767 411 2029 11 1092 369` on a GPU, in the production
configuration, on the released UD-IQ1_S checkpoint — **cannot be met while the
CUDA arm mirrors vLLM.** The CUDA arm emits
`11751 13 15767 411 1928 11 628 567`, **5 of 8** ids, indices 0,1,2,3,5
([evidence](../../docs/bench-evidence/qwen4exp-cuda-prefill-divergence-20260902.md)
§2, the token table).
That is a property of how the objective was WRITTEN, not a measure of how far
the work has got, and it does not soften into "not yet done": the objective asks
the faithful arm to reproduce the unfaithful one.

**RE-TESTED 2026-09-04, AND THE POSITION HOLDS WITH ITS REASON CHANGED**
([ARMTOKENS evidence](../../docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904.md),
[#2858](https://github.com/mudler/vllm.cpp/issues/2858)). The sentence above
rests on the CPU arm being the unfaithful one. **Since [#2612](https://github.com/mudler/vllm.cpp/issues/2612)
it is not**: the CPU GDN prefill runs vLLM's chunked decomposition too, by
default, and that is measured rather than assumed — `VT_GDN_CHUNKED` moves
decoder layer 0's block output by `3.702e-04` on one binary, and the CPU arm
lands `1.772e-05` from CUDA where the sequential arm sat `3.525e-04` away.
**Both arms now mirror vLLM and they still emit the same two sequences, still
5 of 8**, byte-for-byte what PREFILLDIV published. So the objective is
unreachable for a stronger reason than the one written above: it is not that one
arm is faithful and the other is not. It is that two arms running the same
algorithm on the same weights, differing only in device arithmetic, do not agree
on an argmax over near-ties — and with the Gated DeltaNet term spent, what
carries the remaining 3 ids is the undiagnosed MoE residue, which improved
nothing (`moe` `1.269e-04` -> `2.289e-04` from an input 11.6x closer). A fifth
wave must not re-derive this either.

**ANNOTATION 2026-09-04 — THE RATIOS IN THE TWO PARAGRAPHS ABOVE ARE WITHDRAWN,
IN BOTH DIRECTIONS** ([#2877](https://github.com/mudler/vllm.cpp/issues/2877),
full reasoning, both framings and the committed control in [the ARMTOKENS evidence
file](../../docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904.md)). The
sentences are kept so the shape of the error stays visible. The replacement claim
is NOT "the residue did not grow" — that is the same overreach with its sign
flipped. **(a) THE ONE THAT SURVIVES, AND IT STRENGTHENS THIS SECTION'S OWN
POSITION.** `LayerFp` returns early on `s.step >= s.budget`
(`src/vllm/model_executor/models/qwen4_exp_forward.cpp:118`), so
`VT_Q4EXP_LAYER_FP=3` covers forwards 0, 1, 2 = tokens `11751 13 15767`, which
**agree on both arms**, while the three disagreeing ids are emitted at forwards 4,
6 and 7. **No `VT_Q4EXP_LAYER_FP` fingerprint has yet observed a disagreeing
step**, so the attribution of "the remaining 3 ids" to the MoE residue is not
measured here at all. Those forwards are causally upstream of forward 4 through
the recurrent state, so the taps are not irrelevant to the disagreement. They
simply never observe it. **CORRECTION
([#2969](https://github.com/mudler/vllm.cpp/issues/2969)): this sentence said "no
instrument", and that is wrong for the other tap.** `VT_MOE_SEL_FP` counts MoE
block invocations and not forwards, so its 384-call window does reach forwards 4,
6 and 7. MOEDIV's reading there is VOID, because that run's CUDA arm answered the
degenerate pre-#2550 sequence `11751 271 271 271 271 271 0 0`. What was missing
was a non-degenerate arm, not a wider window. **SECOND CORRECTION 2026-09-06
([#2998](https://github.com/mudler/vllm.cpp/issues/2998)): the non-degenerate arm
has been run and forwards 4, 6 and 7 are read.** They flip 127 of 144 expert
slots, and the four agreeing forwards flip 116 of 192, so **the observation does
not restore the attribution this bullet withdrew** — see §"The layer-0 flip
question is ANSWERED" above for why a flip count separates nothing. **(b)** `rel(sumabs)` is a difference of NORMS, not a norm of
DIFFERENCES, and its under-report is a DISTRIBUTION rather than the `~122x` single
seed draw this line first quoted: over 400 seeds at `n = 12800` the median is
75x (sigma 1e-3) to 140x (sigma 1e-4), p05..p95 34..1500, and at a **fixed**
true divergence two readings differ by a median **2.1x** and **24x** at p95 —
where the `4.64x` span quoted by this annotation's own first version
(`c51c484db`) was six seeds. **(c) One framing, applied to every
tap.** `3.525e-04` and `1.269e-04` are the same PREFILLDIV column, so a comparison
that is mismatched for the MoE row is mismatched for the block row. Read as
algorithm-**matched** CPU-vs-CUDA pairs, `L00 blk` moved **16.7x FURTHER**
(1.062e-06 seq/seq -> 1.772e-05 chunked/chunked) where the paragraph above reads
`3.525e-04` -> `1.772e-05` as a 19.9x improvement, and every other layer-0 tap
moved further too (`s.attn` 1.71x, `mhc.mix` 2.02x, `moe` 3.15x, `s.mlp` 2.34x,
`out` 2.29x; `mhc.inj` alone to exactly zero). Against (b), 19.9x and 16.7x sit at
6% and 7% of the metric's own no-change distribution, and no change at all
produces 1.80x, 2.02x, 2.34x and 3.15x in **59%, 52%, 45% and 33%** of draws —
this line first called that "between its 33rd and 59th percentile", which states
the complement and inverts the ranking: 3.15x sits at the **67th** percentile of
no change, not the 33rd. So **nothing here is ranked, in either
direction**. The direction (c) reports is what the chunked decomposition's larger
reassociation freedom predicts — it lands `2.29e-04` from the exact answer where
sequential lands `1.15e-08` — and it is not a defect. The residue's mechanism was
already named by
[#2552](https://github.com/mudler/vllm.cpp/issues/2552) — the keep-quant grouped
expert GEMM's reassociation plus a bimodal top-k term at a 32.9% exact-tie rate,
both floors, both faithful mirrors and neither a defect. **STILL NOT CLOSED, for
a narrower reason since 2026-09-06** ([#2998](https://github.com/mudler/vllm.cpp/issues/2998),
[the reading](../../docs/bench-evidence/qwen4exp-moe-selection-fwd-20260906.md)):
this sentence said the matched pair's `4.324e-05` lands inside #2552's layer-0
flip bracket (2.139e-05 no flip .. 4.999e-04 flip) and that `VT_MOE_SEL_FP` was
never run on it. **It has been run: layer 0 does NOT flip, so the no-flip bound
lifts to `4.324e-05`, and the layer-0 residue at this pair is the keep-quant
expert GEMM's reassociation rather than the top-k term.** What remains open is the
cause of the three disagreeing ids, and this reading does not supply it: selection
flipping is pervasive on the agreeing forwards too, and forward 5 flips 48 of 48
slots while its id agrees. Owed under
[#2999](https://github.com/mudler/vllm.cpp/issues/2999); it needs the 68 GB
artifact and a GPU.

**Why this section exists at all.** The *cause* was already recorded (§Wave
PREFILLDIV, and the evidence file above). The two dead-end *routes* were not, and
a search of `.agents/` and `docs/` for the measured `3 of 8` result returns one
evidence line and no spec position. Three separate waves each re-derived the same
conclusion. A fourth must not.

**1. The two arms run different algorithms, by design.** The CUDA arm runs
vLLM's chunked WY decomposition (`src/vt/cuda/cuda_gdn.cu:6117`
`GdnPrefillChunkedCuda`, selected by the predicate at `:6218`); the CPU arm runs
an exact sequential recurrence (`src/vt/cpu/cpu_ops.cpp:1863` `GdnPrefillKernel`
-> `:1812` `GdnHeadTokenStep`). vLLM runs the chunked one for **all** GDN
prefill: `QwenGatedDeltaNetAttention._forward_core` dispatches
`self.chunk_gated_delta_rule` with no sequential branch
(`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1436-1451`), and
`forward_native` (`:267-295`) returns `fla_chunk_gated_delta_rule` rather than a
torch fallback. So the CUDA arm mirrors upstream and the CPU arm does not
([`gdn-chunked-mirror.md`](gdn-chunked-mirror.md) §Scope,
[#2612](https://github.com/mudler/vllm.cpp/issues/2612)).

**Upstream's CPU arm is chunked TOO, and no record in this tree said so.** The
same layer selects `forward_cpu` on a CPU platform (`:376-382`), which calls
`torch.ops.vllm.cpu_gdn_attention_core` (`:971`), whose prefill is
`ops.chunk_gated_delta_rule_cpu`
(`vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py:646`, also `:241`,
`:589`; binding at `vllm/_custom_ops.py:3359`). vLLM ships a **chunked CPU GDN
kernel**. Our sequential recurrence therefore mirrors upstream on *no* device,
and "upstream has no CPU path, so ours is unconstrained" is not available as a
defence. Read in the local checkout at `5559679229` — see the pin caveat below.

**2. Route A, make the GPU sequential: MEASURED, and it is WORSE on the counted
axis.** `VT_GDN_CHUNKED=0` routes CUDA to `GdnScanCuda` (`cuda_gdn.cu:6223`),
the same sequential recurrence. On the identical binary, prompt and artifact it
drops the first divergent tap — decoder layer 0's Gated DeltaNet block output —
from `rel = 3.525e-04` to `1.062e-06`, a **332x** reduction, and emits
`11751 13 15767 264 1103 314 5656 321`: **3 of 8**, indices 0,1,2. Fewer ids from
a 332x closer hidden state
(the same evidence file, §2 third row and §3). Agreement between our arms is an argmax over near-ties and is **not a
monotone function of the distance between them** — a change that improves the
numbers can lose ids. Route A also forfeits the mirror, which is the reason it
would be refused even if it had won.

**3. Route B, make the CPU chunked: mirror-correct, and it MOVES THE TARGET.**
That repair is specified as `KERNEL-GDN-CHUNKED-MIRROR`
([`gdn-chunked-mirror.md`](gdn-chunked-mirror.md),
[#2612](https://github.com/mudler/vllm.cpp/issues/2612)) and it is the right
change on the mirror criterion. It also **changes the CPU arm's output**: that
spec's own §"The control token sequence" states "Flipping the CPU default to
chunked changes it" and enumerates the fourteen tracked files that carry the
sequence in three different spellings. The specific ids the objective names cease
to exist. There is no ordering of A and B that leaves a target standing.

**4. Therefore, and what WOULD achieve it.** Token identity between the two arms
is achievable only by a CUDA sequential kernel bit-identical to the CPU's, which
means matching its reduction order exactly, element for element. **That is
rejected on two independent grounds**, either sufficient: it is not the
production configuration (it is `VT_GDN_CHUNKED=0`, an A/B rollback flag —
`cuda_gdn.cu:3265-3271`), and it is not a mirror of upstream, which runs the
chunked algorithm on every device it supports. AGENTS.md §"vLLM is the reference"
forbids moving the CUDA arm away from vLLM to make a local arm agree with it.

**5. What the divergence is NOT, so nobody re-chases it.**

- **NOT an unstable CUDA top-k tie-break.** Wave TIEBREAK
  ([#2586](https://github.com/mudler/vllm.cpp/issues/2586), landed as
  [#2595](https://github.com/mudler/vllm.cpp/pull/2595)) measured it on
  `thor:gpu0`: **4652 assertions, 0 failed**, at the `E = 512` `k = 10` geometry
  this model routes, byte-identical to the serial GPU oracle and to the CPU
  reference at an exact bf16 tie, over 279 repeat and 2313 batch-position
  comparisons (§Wave TIEBREAK §Outcome;
  [evidence](../../docs/bench-evidence/qwen4exp-moe-tiebreak-stability-20260902.md)).
- **NOT the chunked reassociation.** In `float64` the chunked decomposition
  reproduces the sequential recurrence to `2.428613e-17` out and `1.110223e-16`
  state — `1e-13` of the gap it was supposed to explain
  ([evidence](../../docs/bench-evidence/gdn-chunked-decomposition-20260902.md), its
  one-line result and the `chunk_f64` row of its table).
- **It is bf16 intermediate storage, and the whole of it.** vLLM's kernel sits
  `2.286426e-04` from the exact recurrence where ours sits `1.15e-08`; a numpy
  replica given nothing but upstream's bf16 placement reproduces the real
  kernel's distance to seven significant figures. At bf16 output width, chunked
  with upstream's placement differs from vLLM's kernel in **62 / 8192** elements
  while both an f32-intermediate chunked port and our sequential arm differ in
  **4157 / 8192** ([`gdn-chunked-mirror.md`](gdn-chunked-mirror.md) §"The
  measurement that decides the design").
- **NOT fully attributed, in one term.** The MoE residue of `7.269e-05` per
  layer that survives with the GDN source removed is named and undiagnosed, and
  it is owed by [#2552](https://github.com/mudler/vllm.cpp/issues/2552) under
  §"PREFILLDIV: the MoE second source" below. It does not reopen 1-4: the layer-0
  tap is bit-identical on input, so the attribution above stands whatever the MoE
  term turns out to be.

**The pin caveat, and it is real.** Every upstream anchor above was read in the
local checkout at `5559679229` — the **prior** parity pin. The pin advanced to
`e126687a9a` on 2026-09-03 ([#2817](https://github.com/mudler/vllm.cpp/issues/2817)),
`qwen_gdn_linear_attn.py` changed across that window, and the layer gained a
`_resolve_gdn_prefill_backend` whose arms are `triton` and `cutedsl`
([`../sync/2026-09-03-e126687-step6.md`](../sync/2026-09-03-e126687-step6.md)
§3.2 item 2)
— both chunked, so nothing above is expected to move. **Expected is not measured.**
Re-anchoring this section's five upstream citations at `e126687a9a` is owed to
[#2831](https://github.com/mudler/vllm.cpp/issues/2831), and no wave should quote
them as "at the pin" until that runs.

**What is owed is a WELL-POSED objective, and choosing it is the developer's.**
This section retires a question; it proposes no plan. Three candidate
replacements have been named in flow and **none is decided**:

| candidate | what it would gate | what it costs |
|---|---|---|
| a prefill hidden-state gate | the layer-0 tap against a tolerance, which is continuous and not bimodal | needs a ratified tolerance, and it is a CPU-vs-CUDA gate rather than an oracle gate |
| selection-set agreement with a tie-rate histogram | expert selection as a SET, with the tie population printed beside it | says nothing about ids, and MOEDIV already found a third of the boundaries are exact ties |
| CUDA-vs-vLLM tokens | the faithful arm against the oracle, which is the comparison AGENTS.md actually asks for | needs an upstream that runs `qwen4_exp` on a fleet device, which §"Upstream's own QSA cannot launch on `thor:gpu0`" ([#2626](https://github.com/mudler/vllm.cpp/issues/2626)) says it cannot |

[#2547](https://github.com/mudler/vllm.cpp/issues/2547) is left OPEN. Its
measurement is accurate and its named candidate was falsified by PREFILLDIV, but
whether the objective it serves is withdrawn or replaced is a developer decision
and not a records repair.

**`docs/USAGE.md` is already corrected and needs no further edit here.** The
recommendation that used to read "`--device cpu` is the arm to use when the exact
ids matter" is gone from the tree; `docs/USAGE.md:893` now reads "**`--device
cuda` is the arm that MIRRORS vLLM; `--device cpu` is the arm that is more
accurate**", with the two figures and both evidence links. The *further* rewrite
— re-deriving the ids that cell records once both arms are chunked — belongs to
[#2612](https://github.com/mudler/vllm.cpp/issues/2612) and is listed in
[`gdn-chunked-mirror.md`](gdn-chunked-mirror.md) `## Records owed`, not here.
That table still anchors the cell at `docs/USAGE.md:670`; the corrected text is
at `:893` today, so the anchor drifted after it landed and the row that repairs
it owns the fix.

### Upstream's own QSA cannot launch on `thor:gpu0` (#2626)

Found by wave RUNHALF while measuring the run half of the `e126687a9a`
candidate ([#2611](https://github.com/mudler/vllm.cpp/issues/2611),
[`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md)
§6). **Upstream vLLM at that revision, built from source for sm_110, cannot run
`Qwen4ExpForConditionalGeneration` on `thor:gpu0`**: its QSA indexer's
`cooperative_topk` refuses to launch with
`a kernel launch error has occurred due to cluster misconfiguration` at
`csrc/libtorch_stable/cooperative_topk.cu:46`. The cluster size there is a
compile-time template parameter, and the arm is chosen by
`get_device_prop()->major >= 10`, a compute-capability major used as a proxy for
a cluster-size capability, which admits Thor's 11.0.

**The obvious hypothesis was tested and NOT confirmed.** A second leg with a
larger request batch, meant to select the 8-CTA arm, failed identically;
`num_rows` at that call site is the number of scoring rows rather than the
request batch, so the knob did not move the variable. Which cluster size was
attempted is unknown, and Thor's actual maximum cluster size was never queried.

**Why this row owns it rather than the sync row.** It is a property of this
model on this hardware and it outlives any pin: `UPSTREAM-SYNC-HEADPIN` closes
when its wave closes, and this would then have no live owner.
[#2626](https://github.com/mudler/vllm.cpp/issues/2626) owns it, and its item 3
is the one that matters here — if upstream cannot run `qwen4_exp` on the only
fleet device whose container exposes a GPU, this row has no vLLM oracle for its
own model on this fleet regardless of the pin, and the `llama-cpp-qwen4exp`
secondary oracle carries more weight than the records currently say.

### PREFILLDIV: the MoE second source, and the instrument it needs (#2552)

Wave PREFILLDIV isolated TWO sources of CPU/CUDA prefill divergence at decoder
layer 0, the only layer whose input is bit-identical on both CUDA arms. The
first is the chunked Gated DeltaNet prefill and it is NOT a defect: it mirrors
vLLM's own Flash-Linear-Attention precision map at every anchor (evidence file
§4). The second is the MoE block, which turns a `2.1e-05` input difference into
a `7.269e-05` output difference and which `VT_GDN_CHUNKED=0` barely moves
(1.7x, against 332x on the Gated DeltaNet tap).

**It lands NOT DIAGNOSED and it is owed.** `VT_Q4EXP_LAYER_FP` taps values, and
a value tap cannot separate an expert-selection flip from re-association in
`vt::MatmulBTQuantGrouped`. The instrument that can is a tap on the SELECTED
EXPERT IDS per token per layer, asserting set equality and printing the margin
between the last selected and the first rejected logit — a discrete selection
has bimodal error, not a tolerance.
[#2552](https://github.com/mudler/vllm.cpp/issues/2552) owns it.

**Also owed by this wave, and smaller.** The `q4fp` line is a `sum|x|`
aggregate. It is coarse enough that it read `0.000e+00` for
`vt::Qwen4ExpGatedResidual`, which is the correct answer at this operand
distribution and is not a proof of bit-identity element by element. A tap that
needs to claim bit-identity should compare bytes, not an aggregate.

- **THE `device_token_ids` CONTRACT IS ADVISORY, AND FIVE MORE ARCHITECTURES
  IGNORE IT ([#2544](https://github.com/mudler/vllm.cpp/issues/2544)).** DECODEDIV
  fixed `qwen4_exp`'s half of this and did not fix the general one. The runner
  assigns `forward_input.device_token_ids` unconditionally whenever the async
  mirror is engaged — which is the default on CUDA — while the field's own comment
  still claims "a model that ignores it is simply never given one". A forward that
  ignores it embeds a host array the runner never wrote for a decode row, and
  `token_ids_cpu` is zero-initialised, so the model generates from token id 0 for
  ever. That is [#1305](https://github.com/mudler/vllm.cpp/issues/1305) and #2496,
  twice, and `grep -rl 'device_token_ids\|DeviceTokenIds'` finds no translation
  unit for `DeepseekV4ForCausalLM`, `Glm5NextForConditionalGeneration`,
  `glm_moe_dsa`, `Laguna` or `KimiK3`. Each is a CANDIDATE and not a conviction —
  the issue is filed on a grep and says so — and convicting one needs the same
  measurement, which needs its checkpoint and its GPU time. The item this row
  really owes is not five ports: it is making the contract ENFORCEABLE, so a
  forward that ignores a live `device_token_ids` is refused by name instead of
  silently decoding from the previous step's identifiers. `VT_ASYNC_RUNNER=0` is
  the same-binary rollback meanwhile.

- **THE N-GRAM HASH IS A HOST COMPUTATION, AND THAT IS WHAT COSTS A DRAIN PER
  STEP (#2496).** `ForwardQwen4ExpForConditionalGeneration` now materialises the
  device identifiers on the host, because `RunQwen4ExpPleBlock` hashes the raw ids
  in host int64 arithmetic (`qwen4_exp::BuildNGramIds`, the port of transformers
  5.16.0 `_splitmix64`, :979-983) and advances the n-gram context the next step
  reads. `detail::ApplyDeviceTokenIds` would repair the embed alone and leave the
  hash reading zeros, so the copy plus drain is forced rather than chosen. It is
  the synchronise ENG-ASYNC-SCHED W4 removed, paid once per step on this
  architecture. Upstream does not pay it: `vllm/models/qwen4_exp/nvidia/ple_layer.py`
  computes the ids in a device custom op, `torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids`,
  over a device `ngram_context` (read at `191cecd51e`, a FORWARD reference past
  this row's pin at `5559679229`). Porting that op is what buys the drain back,
  and it is a wave, not a line.

- **WHAT THE 2026-08-31 vLLM LANDING OWES THIS ROW (Q4RECONCILE, #2489).** The
  oracle reconciliation is done and the divergences are classified in
  `### Component-by-component reconciliation`. What it did not do is act on any of
  them, and each item below needs its own wave, its own issue and its own gate.
  They are listed so the next reader inherits a list rather than a rediscovery.

  - **The f32 QSA query into `vt::RmsNorm` (verdict (a),
    [#2477](https://github.com/mudler/vllm.cpp/issues/2477)).** Upstream's norm
    input is bf16 and this tree's is f32 at exactly one site,
    `qwen4_exp_qsa_block.cpp:705`. Owned by the wave on #2477.
  - **The keep-quant hyper-connection arm (verdict (b),
    [#2406](https://github.com/mudler/vllm.cpp/issues/2406)).** vLLM's HC
    projections are unquantized bf16 by explicit construction. Owned by #2406.
  - **The merged HC down-plus-inject GEMM (verdict (b)).** Upstream stacks them
    into one padded `MergedColumnParallelLinear`; this tree runs three GEMMs. It
    is a `vt::MergedGemmGroup` seam question and IS NOW FILED, row-owned under
    `.agents/issues/MODEL-MM-QWEN4-EXP/`. Its stable local ID is named in
    [the ROCm grouped-norm spec](qwen4-exp-rocm-hc-norm.md) under
    `## Adjacent work, filed and not built`, and NOT here: this heading is the
    ROWLESS registry, and `scripts/check-agent-record.py` reads a row-owned
    stable ID beneath it as an owed reference it cannot resolve.
  - **The deferred, norm-fused HC combine (verdict (b)).** Upstream reads the
    residual once; this tree reads it twice. ALSO FILED, row-owned, with its
    stable ID in the same place and for the same reason. It is the LARGER of the
    two hyper-connection levers, and the ROCm grouped-norm launch-shape fix does
    NOT reach it: that one changes how the norm reads the stream, not how many
    times the stream is read.
  - **The f32 QSA output gate (verdict (b)).** Upstream sigmoids a bf16 gate. It
    rides with #2477 because it is the same buffer.
  - **The single indexer side cache and the every-step re-pool (verdict (b)).**
    Upstream keeps a raw ring plus a persistent compressed cache, both bf16 by
    construction; this tree keeps one cache at `ResolveKvCacheDType()` and rebuilds
    the pooled keys every step. NO issue yet, and the cost grows with context.
  - **The mRoPE position tail on the raw side cache (verdict (c) today, (a) on
    the image path).** Upstream stores the exact three-axis position beside each
    raw key. This tree forces the three axes equal and ropes at `b * CR`, which is
    correct only while they are equal. It becomes wrong the moment the image or
    video path lands, and that path is already owed.
  - **MTP.** `Qwen4ExpMTP` is registered upstream and implemented at
    `nvidia/mtp.py`; this tree parses `mtp_num_hidden_layers` and consumes it
    nowhere. Previously owed against `qwen3_5_mtp.py` as the nearest form; it is
    now owed against a real upstream implementation of this model's own drafter.
  - **The decode GEMM plan table.** `nvidia/low_latency_gemm.py` carries measured
    skinny-GEMM configurations keyed on this model's shapes. It is a lever, not a
    gap, and it cannot be evaluated until a device exists.
  - **Three things this pass could not determine** and which are therefore still
    open questions rather than agreements: the top-k tie semantics, the recurrent
    KV group against upstream's two new state classmethods, and the vision path.
    See `### What this pass could NOT determine`.

- **WHAT #2396 CHANGES FOR THIS ROW, AND WHAT STILL BLOCKS A CUDA FORWARD AFTER
  IT. TRACED IN THE CODE, NOT PREDICTED FROM THE PULL REQUEST.** #2396 adds
  `kEmbeddingQuant`, gives `EmbeddingKernelCuda` a block-decoding arm and makes
  `DeviceQuantGatherSupported` ask `OpRegistered(kEmbeddingQuant, dev)`. Three
  clauses this wave wrote then become FALSE and are listed here so whichever
  branch lands second repairs them rather than discovering them: "the CUDA
  gather arm is owed", "`DeviceQuantGatherSupported` is `kCPU`-only", and "there
  is no `kEmbeddingQuant` OpId". They appear in the PR body,
  `tests/vllm/models/test_qwen4_exp_cuda.cpp` (the SCOPE paragraph) and in this
  spec.

  **OUTCOME, RECORDED BECAUSE THE PREDICTION WAS ONLY TWO-THIRDS RIGHT.** Both
  merged within an hour: #2391 (`27aaf199a`) then #2396 (`f937e8063`). KGATHER's
  author reconciled `docs/FEATURES.md` -- the row now reads "ALL SIX ... NOW HAVE
  CUDA ARMS ... **THAT LEAVES THE GATHER, AND KGATHER LANDED IT**", one story
  rather than two, and the contradiction this entry warned about did NOT survive
  -- and repaired the SCOPE paragraph in `test_qwen4_exp_cuda.cpp`. **The spec
  was the location that was missed**: three live claims in
  `### The neighbouring suites, and the LOAD-TIME refusal that settles
  reachability` survived on `main` stating a gap that KGATHER had closed, one of
  them with a literal `grep -c` that had changed answer. Repaired above by the
  wave that wrote them, tracked by
  [#2443](https://github.com/mudler/vllm.cpp/issues/2443). Nothing is
  outstanding.

  **THE MODEL STILL WILL NOT RUN A CUDA FORWARD, AND THE REASON MOVES FROM THE
  LOADER INTO THE QSA BLOCK.** `qwen4_exp_forward.cpp` builds its rope tables as
  `DBuf rope_packed/rope_cos/rope_sin`, and `DBuf`'s constructor ends
  `t_ = MakeTensor(p_, dt, d.q.device, shape)` -- so on a CUDA queue those
  tensors carry the CUDA device. `QsaBlockCore` passes them straight to
  `CheckRopeLayoutsAgree`, which is `VT_CHECK(cos_sin/cos/sin .device.type ==
  kCPU)`. **That refuses by name on every CUDA forward**, before any token. Two
  more host reads sit behind it: `IndexerRows` dereferences the indexer block
  table on the host, and `Qwen4ExpQsaIndex` dereferences `kv_lens` on the host to
  build the scoring window. All three are `qwen4_exp_qsa_block.cpp`'s own owed
  item -- this row's, not #2396's -- and each already says so in its refusal
  string.

  **SO THE HONEST POST-MERGE STATEMENT IS NEITHER "VACUOUS" NOR "RUNS".** The
  decoder layer runs PLE first (`qwen4_exp_forward.cpp:392`), then GDN or QSA
  (`:431`/`:440`), then MoE (`:498`). With #2396 and this wave both landed, a
  CUDA forward is EXPECTED to dispatch the gather, `vt::RmsNormGroup`,
  `vt::Qwen4ExpPleGate`, `vt::Qwen4ExpPleConv`, `vt::Qwen4ExpGatedResidual` and
  `vt::Qwen4ExpGatedResidualWriteBack` -- five of this row's seven CUDA arms plus
  the new gather -- and then to refuse at the first QSA layer. **THAT IS A
  PREDICTION FROM READING THE CODE AND IT IS NOT MEASURED.** It is written down
  so it can be tested rather than assumed: the test is a `--device cuda`
  `ModelRegistry::Forward` on the synthetic fixture, and the expected result is
  the `CheckRopeLayoutsAgree` refusal string, not a token. Until that runs, this
  row claims no reach beyond what M8 measures.

  **AND A SECOND, INDEPENDENT BLOCKER APPLIES TO THE RELEASED ARTIFACT ONLY.**
  `IsCudaKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu`) admits twelve
  dtypes -- IQ2_XXS, IQ3_XXS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ1_S,
  IQ1_XXXS, IQ2_XS, IQ4_XS -- and **not IQ4_NL, Q5_0, Q4_0 or Q8_0**. The shipped
  `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S uses IQ4_NL and Q5_0, which this
  tree added for exactly that file, so its GEMM path would drain to the host or
  refuse on CUDA even with the QSA host reads moved. The synthetic fixture does
  not hit this; the real checkpoint does. Owed, and not this wave's.

- **THE `docs/FEATURES.md` CONFLICT WITH #2396 IS A TAKE-BOTH HAZARD, AND THE
  CONTRADICTION IS EXACT.** Both branches edit the `Qwen4ExpForConditionalGeneration`
  row. #2396's version RETAINS the sentence "no CUDA arm exists for any
  `qwen4_exp` op, so there is no device arm to run" and ADDS "several `qwen4_exp`
  ops still have no CUDA arm"; this wave's version replaces the first with "ALL
  SIX `qwen4_exp` ops PLUS `vt::RmsNormGroup` NOW HAVE CUDA ARMS". An automatic
  take-both puts both readings in ONE table cell, where the older one is simply
  wrong. The merged form must carry #2396's load-side fact (the gather arm
  exists, `DeviceQuantGatherSupported` says yes on CUDA, METAL/VULKAN/ROCM/
  TENSTORRENT still refused), this wave's op-side fact (all seven arms exist),
  and the reconciliation NEITHER branch contains: what now blocks a CUDA forward
  is the QSA block's three host reads, not op registration and not the loader.

- **A CI DEFECT FOUND IN FLOW, AND MAIN FIXED IT BETTER. ISSUE-GH-2407 IS A DUPLICATE.**
  `agent-record` failed on EVERY pull request: `.github/workflows/ci.yml` still
  ran `tests/scripts/test_check_issue_index_append_only.py`, which `7dc2ef1ea`
  deleted when it retired the append-only index, so the job printed
  `agent record OK: ENGINE=173 MODEL=379 ...` and then exited 2 on a missing
  file -- the gate green, the check red. This wave filed
  ISSUE-GH-2407 and DELETED the line.
  **`origin/main` `df2ad6d84` ([#2371](https://github.com/mudler/vllm.cpp/issues/2371),
  #2373) had independently REDIRECTED it instead**, to
  `tests/scripts/test_agent_issue_index.py`.
  **Main's is the better fix and this row took it wholesale**, which is the
  honest call rather than the flattering one: deleting the invocation drops a
  gate, whereas redirecting it keeps one, and the file main points at tests
  `scripts/agent-issue-index.py` -- the derived renderer that REPLACED the
  retired index. So main's fix is BROADER than this wave's, not narrower. The
  merged `ci.yml` is byte-identical to main's, this branch carries no CI change,
  and ISSUE-GH-2407 closes as a duplicate of #2371 rather than as work this row did.

- **W6-CUDA-B LANDS FOUR CUDA ARMS THAT NOTHING REACHES, AND ONE BLOCKER
  REMAINS. ISSUE OWED.** `vt::Qwen4ExpGatedResidual`, `vt::RmsNormGroup`,
  `vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention` now have CUDA
  arms, so ALL SIX `qwen4_exp` ops plus `vt::RmsNormGroup` do. **They land
  UNREACHED** under AGENTS.md "Nothing lands dead": `ModelRegistry::Forward` is
  all-or-nothing and `EmbeddingKernelCuda` refuses a block-quantized table by
  name, so the block-decoding n-gram gather has no CUDA arm and no `qwen4_exp`
  step can reach a CUDA queue. The wiring is owned by row `MODEL-MM-QWEN4-EXP`
  under campaign [#1978](https://github.com/mudler/vllm.cpp/issues/1978),
  tracked by [#2031](https://github.com/mudler/vllm.cpp/issues/2031) and by
  this wave's own ISSUE-GH-2380.
  **THAT ISSUE EXISTS, WHICH THE PREVIOUS FIVE WAVES ON THIS ROW COULD NOT
  MANAGE, AND THE REASON IS WORTH RECORDING.** Every entry above says the `gh`
  token is invalid on this host and cites `gh api user` returning 403. On
  2026-08-31 it returns **200** for `localai-org-maint-bot` and `gh issue create`
  succeeds. What is still true is the OTHER half of that story: `gh issue view
  2031` returns "Could not resolve to an issue", and ISSUE-GH-2379 explains why — those
  issues were created by an account that has since been suspended, and GitHub
  HIDES a suspended account's content rather than deleting it. So the numbers
  this spec cites dangle for a reader even though the API is writable again.
  A 403 on writes and a hidden issue on reads are different failures, they were
  conflated on this row, and only the second one is still live. Separately,
  `IsCudaKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu`, the function of that
  name -- a LINE RANGE was written here first and `origin/main` `c2daafb23`
  invalidated it inside this very branch, which is the drift this file keeps
  recording) returns
  false for IQ4_NL, Q4_0, Q5_0 and Q8_0, so the mixer's quantized projection
  route through `vt::MatmulBT` refuses or drains to the host on a CUDA queue —
  also another wave's, also owed. See `### W6-CUDA-B` below.

- **TWO SPEED ITEMS THIS WAVE DECLINED, EACH WITH ITS CONDITION. ITEM (1) IS
  HALF DISCHARGED BY W9 AND ITEM (2) IS LIVE.** (1) The gather's pass-2 dot WAS
  SEQUENTIAL ON ONE LANE, `|sel| * head_dim` dependent f32 operations, ~525k per
  (token, head) at the released config. **W9 removed that cost WITHOUT taking the
  tree reduction** (`7d0d74c2c`): pass 2's tile now goes ONE WHOLE DOT PER
  THREAD, which reassociates nothing, because a dot's value does not depend on
  which thread evaluates it. The red-first measurement this entry asked for is
  the attribution the row's kernel-time issue carries — that lane measured 86.4%
  and 88.8% of the kernel's wall time in two `thor:gpu0` runs — and the change
  measured 6.52x at the released decode shape. (The issue is deliberately NOT
  cited by ID here: `scripts/issue_records.py:518` refuses a row-owned issue that
  a spec's `## Owed` names, because that is the shape reserved for a rowless
  `_owed` record.)
  **What remains owed is the lever itself, still declined:** a deterministic tree
  reduction over `d` would preserve the gather-vs-dense bit-identity — the order
  would depend on `head_dim` alone — while breaking the CPU-vs-CUDA relation the
  sequential dot keeps. It is now a SECOND-ORDER item rather than the
  bottleneck, and the condition is unchanged: take it only after a red-first
  measurement that shows the per-thread dot is the bottleneck. The fold over `s`
  that W9 preserved is no longer only argued: `test_qwen4_exp_cuda_reductions.cpp`
  refolds the arm's own softmax weights ascending on the host and requires BIT
  equality with the arm's denominator, which is red against a descending fold and
  against the warp-shuffle tree. (2) The mixer takes ONE `cudaMalloc` per call
  for its four intermediates. A static cache would not be re-entrant; the fix is
  a caller-supplied workspace, which changes the op signature and owes its own
  spec. **LIVE**, and tracked by its own row-owned issue under
  `.agents/issues/MODEL-MM-QWEN4-EXP/`, unnamed here for the same reason.

- **W8CONFIRM ISOLATED THE CAUSE TO W5r's TWO LINES, WHICH W5s ASSERTED BUT ITS
  EVIDENCE COULD NOT SEPARATE FROM W5p. ISSUE OWED.** W5s compares W7DIAG on
  `701606e51` against itself on `52f7ccbfc`; that span contains W5p **and** W5r,
  so it cannot apportion the repair, and its `VT_CPU_QUANT_REPACK=0` arm returns
  no signal at all because with the fix present both arms are correct and come
  back bit-identical. W8CONFIRM builds TWO binaries from ONE tarball of
  `52f7ccbfc` differing only in `dense_attn_block.h:235-236`, runs both against
  ONE staged copy in ONE lease, and gets `X-ON` (fix reverted, repack ON)
  `"!!!!!!!!!!!!!!!!"` against `X-OFF` (same binary, repack OFF) and both `M`
  arms coherent. Same binary either side of one environment variable, so the
  defect needs the chain ACTIVE and the markers DROPPED; W5p is in all four arms
  and cannot explain a difference between them. See
  `## Mutation record — W8CONFIRM`. **What is still owed:** W8CONFIRM's own
  issue, unfilable because the `gh` token is invalid on this host (`gh api user`
  is 403) while `git push` over SSH works.

- **THE DISCRIMINATOR THIS SECTION PRESCRIBES IS UNDER-SPECIFIED, AND IT WAS RUN
  IN THE ONE CONFIGURATION WHERE IT CANNOT DISCRIMINATE.** The W5r entry below
  says "re-run W5q's exact request on a `thor` lease with
  `VT_CPU_QUANT_REPACK=0` ... If the output stops being constant, this was it."
  That test only has signal on the PRE-FIX tree, where arm A is still degenerate.
  Run on a tree that already carries W5r — which is what W5s did — neither arm is
  constant, the premise never holds, and A ≡ B is a confirmation of the
  performance-transform invariant rather than evidence about the cause. The
  prescription should have named the TREE as well as the flag. Recorded because
  the next person to reach for a one-flag A/B on a fixed tree will get a
  confident null result, which is the failure mode this file exists to prevent.

- **`docs/USAGE.md` OWES A WEIGHTS ROW FOR THIS ARM, AND IT IS NOW DUE.** The
  spec has said the row lands "in the same change that makes an arm SERVE".
  W5s and W8CONFIRM together make it serve coherently on `--device cpu`, so
  AGENTS.md "Say which weights, and from where" is now live for
  `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S: file names, 67.564 GiB over three
  shards, shard1 sha256
  `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`, and the
  refused arms beside it. Neither wave lands product code, so neither carries it;
  it is owed by the next wave that touches this arm.

- **THE `q8_0_aligned` MARKER IS STILL DROPPED BY THE SHARED HELPER, AND
  W8CONFIRM DOES NOT CHANGE THAT.** It is read only in
  `src/vt/cuda/cuda_quant_dot.cu`, so it is inert on every `--device cpu` arm
  measured here and none of these four arms can see it. The CPU GEMM keys on
  `b.repacked` alone (`cpu_quant_gemm.cpp:156`), which is why the two lines W5r
  restored were sufficient for this path and are NOT sufficient evidence for the
  device path. Unchanged from W5r's own entry; restated because a reader who sees
  "the cause is isolated" could conclude the marker family is closed.

- **A JOB THAT PRINTS `cleanup done` CAN STILL HOLD THE FLEET DEVICE, AND THE
  LAG IS NOT ALWAYS THE HANG.** W5s printed every terminal marker, went
  byte-static, and `rc` still listed it running until an operator killed it;
  thor then picked up the next job in 14 s. W8CONFIRM printed `cleanup done` at
  06:52:24 and `rc devices` still showed it busy at 06:54:11, then released on
  its own — roughly 110 s of teardown lag with no intervention. The two look
  identical for the first two minutes, so "still listed" is not by itself the
  known hang. **AND BYTE-STATIC OUTPUT DOES NOT SEPARATE THEM EITHER**, which is
  the trap worth writing down: a job in ordinary teardown has ALSO stopped
  writing, so the check that feels like proof is necessary and not sufficient. It
  rules out "still working"; it does not rule in "hung". W8CONFIRM was called a
  fourth occurrence of the hang on exactly that reasoning and it was wrong — the
  device released itself ~110 s later with no intervention. Confirm byte-static,
  then WAIT past two minutes, and never kill another session's job on the
  strength of the listing alone.

- **W5s SETTLED THE ROW'S BLOCKER: THE RELEASED CHECKPOINT NOW EMITS REAL
  TOKENS, AND THE CAUSE WAS W5r's DROPPED REPACK MARKER. ISSUE OWED.** On
  `origin/main` `52f7ccbfc` — W5p *and* W5r — the artifact answers two different
  prompts with two different, correct, prompt-dependent completions on
  `thor:gpu0`, `--device cpu`:

  ```text
  prompt "The capital of France is"  (prompt_tokens=5) -> " Paris. Given this fact, what is"
  prompt "Water boils at"            (prompt_tokens=3) -> " 100°C at sea level"
  ```

  Eight distinct token ids (`11751, 13, 15767, 411, 2029, 11, 1092, 369`), none
  of them 0, against W5q's eight consecutive id 0 on the same box and artifact.
  Evidence:
  [`docs/bench-evidence/qwen4exp-released-checkpoint-tokens-20260831.md`](../../docs/bench-evidence/qwen4exp-released-checkpoint-tokens-20260831.md).

  **THE MECHANISM IS MEASURED, NOT INFERRED.** Reusing W7DIAG's read-only probe,
  the pre-W5r tree (`701606e51`) and this one agree *numerically* on `embed`
  (l2 0.473868) and `stream.after_widen` (l2 0.947736) and then diverge at
  exactly one place: `stream.after_layer_0` was `nan=51200` and is now `nan=0`,
  and `LOGITS` was `zero=248320` — a row with no maximum, which is why `argmax`
  returned index 0 — and is now a ranked distribution, `min -9.89818
  max 15.7873`, whose argmax id **11751** is the `" Paris"` token that came out
  of HTTP. The identical prefix rows make the pre-fix run a control rather than
  a different experiment.

  **`VT_CPU_QUANT_REPACK=0` IS NOW A CONFIRMATION, NOT A DISCRIMINATOR.** With
  the marker fix in, repack ON and repack OFF give byte-identical text, the
  identical eight ids, and bit-identical probe rows at every stage. That is the
  correct outcome for a performance transform, and it is the thing that was
  false before W5r.

  **WHAT W5s DOES NOT CLAIM.** It is not a token gate: no oracle decoded these
  prompts, and llama.cpp is not a usable same-box control (the sibling wave's
  arm exited 126; it aborts in `build_delta_net_chunking` before loading a
  byte). No speed number — the box carried four other sessions' processes and
  every arm is n=1. Only the UD-IQ1_S arm ran. `num_reqs > 1` is still refused
  by name. **A TOKEN GATE AGAINST AN ORACLE IS THE NEXT OWED STEP.**

- **`logprobs` VALUES ARE NOT PRODUCED FOR THIS MODEL, ON EITHER TREE. ISSUE
  OWED.** W5q owed a `logprobs` request; W5s made it, on both trees, and it
  settles nothing about the logits. The request is accepted and the payload is
  built, but `token_logprobs` and `top_logprobs` come back all-`null`. **That is
  not a serialized NaN.** It is the
  `step == nullptr || step->empty() || step_token == nullptr` branch of
  `BuildCompletionLogProbs` (`serving_utils.cpp:130-135`), which emits the
  `"token_id:N"` fallback string beside the nulls — exactly what both runs show —
  so the engine produced no logprobs dict for those positions.
  `serving_completion.cpp:457` gates the payload on
  `output.logprobs.has_value()`, and that is false here. Why it is false is
  unread. **Anyone reading a `null` logprob on this row as evidence of a NaN
  logit is reading the wrong branch.**

- **`VT_DEBUG_SAMPLED=1` PRINTED NOTHING AND W5s CANNOT SAY WHY. ISSUE OWED.**
  The variable is real (`runner.cpp:3078-3086`), its guard is
  `kDebugSampled && !toks.empty()`, `fprintf(stderr, ...)` is unbuffered and the
  job captured stderr — and all four arms logged zero `vt-debug sampled` lines.
  Recorded as a **failed instrument that contributed nothing**, never as
  evidence about the sampler; the ids above come from the `logprobs` array and
  the decoded text, which do not depend on it. W5s also asserted this variable's
  file and line in its own job script *before verifying them*, and the line
  number was wrong by ~115 lines; the variable's existence was not.

- **W5q's STAGED-COPY RECLAMATION QUESTION IS ANSWERED.** W5q could not say
  whether its 67 GB staged copy was reclaimed, because its `df` ran one line
  after the `rm`. W5s reaped the server first, then `sync`ed, then waited 10 s,
  then measured: `364G used / 508G avail`, byte-identical to the `df` taken
  before the job. It is reclaimed.

- **W5q's `tee` EXIT-TRAP HANG DID NOT RECUR, AND THE LIKELY CAUSE IS NAMED.**
  W5q printed `DONE` and still held `thor` for 42 minutes. W5s dropped
  `exec > >(tee ...)` entirely (rc captures stdout), sent the high-frequency
  sampler to a file, and killed **all four** background children in the trap —
  W5q killed its heartbeat but never its sampler subshell, which inherits the
  job's stdout and is the most likely reason the pipe stayed open. W5s released
  the device on its own, 18 s before the next queued job started.

- **W5r NAMED A SITE FOR THE DEGENERATE OUTPUT BELOW AS A TRACED CANDIDATE, AND
  W5s ABOVE MEASURED IT AND CONFIRMED IT WAS THE CAUSE. KEPT IN THE PAST TENSE
  BECAUSE IT IS THE REASONING THAT FOUND THE DEFECT.** The entry below says
  "W5q identifies no site and no line". W5r identifies one, by reading the
  chain rather than by guessing, and the whole chain is on the box W5q ran on:

  1. `LoadMatmul` routes `hc_*_down` `[320, 10240]` and `hc_*_up` `[10240, 320]`
     to `kKeepQuant`, which calls `OwnGgufQuantBlocks(..., pol.quant_repack)`
     (`qwen4_exp_weights.cpp:265-266`, `:136-137`).
  2. `pol.quant_repack` is `keep_quant && !cpu_ref && vt::cpu::QuantRepackActive()`
     (`gguf_keep_quant.cpp:347`). `VT_CPU_REF` is off by default and the artifact
     is kept quantized, so on an i8mm host this is TRUE. `thor` is aarch64 i8mm.
  3. `QuantRepackEligible` accepts both shapes: Q8_0, `n % 4 == 0` and
     `k % 32 == 0` hold for `320 x 10240` and for `10240 x 320`
     (`cpu_quant_repack.cpp:43-51`). The buffers are rewritten to
     `block_q8_0x4` and `o.repacked = true` (`qwen3_5_gguf_weights.cpp:133`).
  4. The forward takes them through `dense_attn::ResidentWeight`
     (`qwen4_exp_forward.cpp:421-422, :479-480, :538-539`), which **dropped the
     marker** — the defect W5r fixes. `kMatmulBTQuant` then reads
     `block_q8_0x4` bytes as flat `q8_0`.

  That is a wrong mixer on all 48 layers, both sides, plus the terminal mixer,
  with no crash and no refusal — the failure mode that produces a logit row with
  no maximum. **WHAT IS NOT CLAIMED: that this IS the cause.** W5r ran CPU-only
  on x86, where `QuantRepackActive()` is false and the whole chain is inert, so
  it reproduced nothing and fixed nothing observable. Step 2 is the only link
  read off policy rather than measured on that run. The entry below is right
  that three causes sample id 0 and that guessing between them is the error to
  avoid; this is one candidate with a traced mechanism, not a verdict.

  **THAT DISCRIMINATING MEASUREMENT IS DONE — W5s RAN IT.** The prediction in
  this entry held: the output stopped being constant. Step 2, the only link this
  entry read off policy rather than measured, was confirmed on the box
  (`i8mm PRESENT`, so `QuantRepackActive()` is TRUE there). The paired
  `logprobs` request was also made and did NOT settle the NaN / zero / constant
  question, for the reason recorded in its own entry above; the per-stage probe
  settled it instead, and the answer is a NaN born in layer 0 collapsing to an
  all-zero logit row.

- **W5r's FIX IS GATED BY CONSTRUCTION AND HAS NEVER RUN ON A HOST THAT SETS THE
  MARKER. ISSUE OWED.** `dense_attn::ResidentWeight` now carries `repacked` and
  `elem_kn_repacked` from the `OwnedTensor` to the `vt::Tensor` the kernel sees,
  as `qwen3_5.cpp`'s private copy of the same helper always has (:1055, :1060).
  The gate sets the flag BY HAND on the same `OwnedTensor` type the loader
  produces and asserts the helper propagates what it is given. It cannot do
  better on this host: `vt::cpu::QuantRepackActive()` returns false off aarch64
  (`cpu_quant_repack_arm.cpp:275`) and `elem_kn_repack` defaults OFF
  (`gguf_keep_quant.cpp:359`), so on x86 both markers are always false and no
  end-to-end path can set one. What is therefore NOT claimed is that a repacked
  `hc_*_down` produces correct tokens; that needs the `thor` lease above.

  The same fact bounds the RISK in the other direction, which is why the change
  is safe to land unmeasured: 25 models inherit this helper, and on every x86
  host both markers stay false, so the two new lines are inert and no golden can
  move. Behaviour changes only where the loader actually sets a marker, which is
  exactly where the old behaviour was silently wrong.

- **`q8_0_aligned` IS A THIRD MARKER AND THE SHARED HELPER STILL DROPS IT.
  ISSUE OWED.** `OwnedTensor::View()` carries `repacked` AND `q8_0_aligned`
  (`qwen3_5_weights.cpp:497-499`); W5r taught `ResidentWeight` the first and the
  `[K,N]` one, and deliberately did not touch the third. That marker is the CUDA
  coalesced-Q8_0 layout, set at `qwen3_5_gguf_weights.cpp:120`, and it is read on
  the DEVICE arm — which W5r could neither run nor gate, being CPU-only with no
  lease. Propagating it blind would change kernel selection for 25 models against
  no gate at all. It is recorded as a gap rather than guessed at: the audit that
  found two dropped markers found three, and the third is still dropped.

  Note also what W5r did NOT port from the private copy. `qwen3_5.cpp` also
  refuses a `repacked` weight (:1105) and an `expert_streamed` tower (:1085) at
  device staging; only the `elem_kn_repacked` guard came across, because it is
  the one W5r could reach with a test. The other two are tripwires for lanes the
  shared helper's 25 callers do not all have, and adding an ungated refusal to a
  shared path is how a correct guard reds normal work. Same owner, same lease.

- **W5r's ISSUE IS OWED.** The `gh` CLI token on this host is invalid
  (`gh auth status`: "The token in ~/.config/gh/hosts.yml is invalid"), so
  `gh issue create` cannot run, while `git push` over SSH does work — the
  precise split the W5q record already corrected. The work rides #2031, the
  row's own issue, and this entry names the debt until an issue exists.
  `.agents/issue-index.md` is retired to `.agents/completed/`, so no index row
  is owed.

- **THE RELEASED CHECKPOINT SERVED AND ITS OUTPUT WAS DEGENERATE. THIS WAS THE
  ROW'S BLOCKER; W5s ABOVE IDENTIFIED THE CAUSE AND THE OUTPUT IS NO LONGER
  DEGENERATE. KEPT AS THE MEASUREMENT THAT DEFINED THE PROBLEM.** W5q drove
  `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S through `examples/server` on the
  composed W5p + LOAD-IO tree, on `thor:gpu0`, `--device cpu`, staged to
  worker-local disk. **The W5n refusal is gone**: the prefill completes, eight
  decode steps run with `model_executed=1` each, nothing throws, and
  `POST /v1/completions` returns **200** instead of W5n's 500. **And the answer
  is a constant.** Both of these came back byte-identical:

  ```text
  prompt "The capital of France is"  (prompt_tokens=5) -> text "!!!!!!!!"
  prompt "Water boils at"            (prompt_tokens=3) -> text "!!!!!!!!"
  ```

  `!` is **token id 0** in this artifact's own vocabulary — read off shard 1's
  `tokenizer.ggml.tokens` (248,320 entries; ids 0-5 are `!`, `"`, `#`, `$`, `%`,
  `&`), not assumed from another model's tokenizer. Eight id-0 samples that do
  not move with the prompt is what `argmax` returns over a logit row with no
  maximum.

  **WHAT THE NEXT WAVE MUST NOT DO IS GUESS.** This run CANNOT distinguish an
  all-`NaN` logit row, an all-zero row and a constant non-zero row, because
  `logprobs` was not requested and the response carries `null` for it. All three
  sample id 0 and all three have different causes. The cheapest discriminating
  step is one request with `logprobs` on a loaded server, and after that a
  per-layer probe: the gate cannot see this, because every gate on this row
  drives the synthetic ramp fixture and the fixture is green.

  **THE SHAPE IS THE ONE W5p ALREADY NAMED**, one level further in: the
  `## Owed` fixture entry says every `vt::` op this loop composes has a dtype
  contract the fixture exercises on exactly one side, and that the mixer refusal
  was unlikely to be the last. It was not. But naming a class is not naming a
  site, and **W5q identifies no site and no line**. Evidence:
  [`docs/bench-evidence/qwen4exp-released-checkpoint-serve-20260831.md`](../../docs/bench-evidence/qwen4exp-released-checkpoint-serve-20260831.md).

- **THE `tee` EXIT-TRAP HANG SURVIVED ITS OWN FIX, AND HELD `thor` A THIRD TIME.
  ISSUE OWED.** LOAD-IO recorded this shape and its workaround: `exec > >(tee)`
  leaves a `tee` that exits only after the trap, so a bare `wait` in an EXIT trap
  holds the fleet device. W5q's job carries that fix — there is no `wait` in its
  trap — and it STILL held the device. The job logged `=== DONE ===` and
  `cleanup done` at 03:25:55 and `rc devices` reported `thor:gpu0` busy with it
  at 42m6s elapsed. `rc kill <full job id>` freed it at once and the next queued
  job started within 8 s. **So the workaround is not sufficient and the bare
  `wait` was not the whole cause.** Until this is understood, the operational
  rule is the one that actually caught it: after a job prints DONE, read
  `rc devices` and kill the job if the device is still held. A job template that
  does not need `tee` at all — writing the log to a file the submitter can read
  off the share — avoids the construct rather than working around it.

- **W5q CANNOT SAY WHETHER ITS 67 GB STAGED COPY WAS RECLAIMED. ISSUE OWED.**
  The cleanup's `du` measured 69 G under `/tmp/w5q`, `rm -rf` ran, and the `df`
  on the next line — same timestamp — read 431 G used against 364 G before the
  job. The server process appears in `ps` one second earlier at 316% CPU, so the
  `df` was taken before an unlink of that size could settle and possibly while
  the file was still open. This is reported as UNKNOWN rather than resolved
  either way; a staging job should verify reclamation after the process exits,
  not one line after issuing the `rm`.

- **`VT_LOAD_STATS` AND `VT_GGUF_PREFAULT` ARE UNDOCUMENTED IN `docs/USAGE.md`.**
  LOAD-IO made the first one report on the GGUF path and both now change what a
  user sees and how long a load takes — `VT_GGUF_PREFAULT=0` took the same load
  from 60 s to 15 s. Neither appears in `docs/USAGE.md`. Pre-existing at W5q and
  recorded rather than fixed here, because a user-facing knob's documentation is
  a scoped edit and not this wave's.

- **W5p FIXED THE BLOCKER BELOW, AND THE ENTRY IS KEPT IN THE PAST TENSE.** The
  op now takes a block-quantized `mix_down`, `mix_up` and `block_inject` and
  routes each through `vt::MatmulBT`, which dispatches `kMatmulBTQuant`. Of the
  two options this entry laid out, the FIRST was taken: the op grew the arm,
  rather than the loader narrowing its policy and expanding 1.17 GiB. The reason
  is not size — it is that llama.cpp merged this architecture to master on
  2026-08-27 (`6c84c7d5d`, PR #27742, first tag `b10660`) and runs this exact
  file WITHOUT dequantizing, declaring all six projections `GGML_OP_MUL_MAT`
  (`src/llama-arch.cpp:759,760,761,763,764,765`) and `hc_*_norm` `GGML_OP_MUL`
  (`:758`, `:762`). Expanding at load would also be re-triggered by the next
  re-quant: the Q8_0 is unsloth's own `--tensor-type` override and no `hc_` entry
  exists in llama.cpp's `src/llama-quant.cpp` allowlist, so a shape fallback
  would emit IQ4_NL next time. This entry's LAST sentence was heeded: the gate
  builds the fixture tensors QUANTIZED (`FixtureOpts::hc_mix_q8_0`), and mutation
  M1 of the W5p record proves that without it the whole thing would have been
  green for the reason the entry names. **What W5p does NOT claim: nothing had
  run the RELEASED checkpoint through the repaired path AT W5p.** The gates are
  the miniature and the op. **W5q then ran it**: the refusal is gone and the
  output is degenerate — see this section's first entry.

- **THE FORWARD REFUSED THE ONLY PUBLISHED ARTIFACT THAT FITS. W5n MEASURED IT;
  W5p FIXED IT. Kept because it is the measurement, and because the next reader
  needs the cause rather than the verdict. ISSUE OWED.**

  ```text
  vt: qwen4_exp_gated_residual: input_mix_weight_down must be float
  (f32/bf16 for outputs)          at src/vt/ops.cpp:2552
  ```

  Measured on `rc` job `0f188dd1`, `thor:gpu0`, 2026-08-30, on
  `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S loaded through `examples/server`
  at `--device cpu`. The model loads, the engine sizes all three cache groups,
  the server listens, `POST /v1/completions` is accepted, prefill begins at
  `prompt_tokens=5`, and the first forward throws. The throw is inside the
  EngineCore busy loop, so the engine dies and the request returns 500.
  **Zero tokens.** Evidence:
  [`docs/bench-evidence/qwen4exp-released-checkpoint-serve-20260830.md`](../../docs/bench-evidence/qwen4exp-released-checkpoint-serve-20260830.md).

  **The cause is a dtype contract, not a bad checkpoint**, and every step is
  measured:

  1. The published file stores **194** hyper-connection mix weights as **Q8_0**:
     `blk.N.hc_attn_down`, `blk.N.hc_attn_up`, `blk.N.hc_ffn_down`,
     `blk.N.hc_ffn_up` (48 each, `[10240, 320]` and `[320, 10240]`) plus
     `output_hc_down` and `output_hc_up`. The `hc_*_norm` and `hc_*_inject`
     tensors are F32, which is why the norm is not what refused.
  2. `qwen4_exp_weights.cpp:265-266` loads `down`/`up` with
     `LoadMatmul(g, pol, ...)`, which honours the keep-quant policy. Q8_0 has a
     `vec_dot` and 10240 and 320 are multiples of its 32-element block, so
     `RouteGgufTensor` returns `kKeepQuant`. **That decision is correct** — it is
     what keeps this model inside 122.80 GiB.
  3. `vt::Qwen4ExpGatedResidual` accepted float only
     (`src/vt/ops.cpp`, `check_operand`). Since W5p a PROJECTION operand takes
     `check_projection` instead and the three elementwise ones keep
     `check_operand`.
  4. **The fixture could not express the failing case.**
     `tests/support/qwen4_exp_gguf_fixture.h` wrote those same tensor names with
     ggml type `0` (F32). Every green gate on this row had handed that op a
     float operand; the published artifact hands it a block. **No gate on this
     row could have caught this**, which is the same shape as W5b-6's gamma
     polarity: a contradiction that is unreachable while only one side of it is
     ever built. W5p added `FixtureOpts::hc_mix_q8_0`, which is the side that was
     never built.

  **W5n deliberately did NOT fix it, and W5p did.** The wave was scoped to run the released
  artifact and report, and a workaround would have destroyed the measurement. The
  fix is its own wave and needs a design decision this row should not take
  silently: either `vt::Qwen4ExpGatedResidual` grows a quantized-operand arm
  (two `kMatmulBT`-shaped `[10240,320]`/`[320,10240]` products against a Q8_0
  weight, which `vt::` already has kernels for), or the loader narrows the
  policy for the `hc_*_{down,up}` role and expands just these 194 tensors —
  measured cost 2 × 48 × 10240 × 320 × 2 B ≈ **1.17 GiB** of bf16 for the
  per-layer pairs plus 12.5 MiB for the model-level mixer, which fits. The first
  is faster and mirrors what the file asks for; the second is smaller and lands
  in one place. **Neither was chosen here.** A gate for whichever lands must build
  the fixture tensors QUANTIZED, or it will be green for the same reason the
  one current at W5n was. W5p took the first and built that fixture arm.

- **THE REPAIRED PATH IS A PER-TOKEN MATVEC, AND THAT IS A SPEED DEBT W5p DID NOT
  PAY. ISSUE OWED.** `ProjectRow` calls `vt::MatmulBTQuant` with `M = 1` inside
  the kernel's existing per-token loop, so a prefill of `T` tokens makes `T`
  keep-quant GEMM calls per projection where llama.cpp makes ONE over the whole
  batch (`build_lora_mm(w_down, xn)` on the `[hc_dim, n_tokens]` activation,
  `src/models/qwen4exp.cpp:237`). At the released geometry that is
  `T x 3 projections x 2 hyper-connection sites x 48 layers` calls per forward,
  each of which allocates its own activation scratch and enters
  `ParallelForRows`. Batching means hoisting the grouped norm for a TILE of
  tokens into a `[tile, hc*H]` buffer and running one GEMM per projection, which
  moves the fused kernel's loop structure and owes its own red-first
  measurement — so it is a wave, not a follow-up edit. **No number is quoted
  here**: nothing has profiled it, and a cost derived from a call count is an
  arithmetic claim rather than a measurement.

- **`.agents/oracles/llama-cpp-qwen4exp.md` IS NOW STALE IN ITS CENTRAL CLAIM.
  ISSUE OWED. DO NOT READ THIS AS A LICENCE TO ADVANCE THE PIN.** That file's
  title says "the only llama.cpp that knows `qwen4exp`" and its evidence table
  records "the PR is unmerged" and "no released llama.cpp has it". Both stopped
  being true on 2026-08-27. Re-derived on 2026-08-31 from the local
  `ggml-org/llama.cpp` checkout's fetched `origin/master`, not relayed:

  | Claim | Command | Result |
  |---|---|---|
  | #27742 is MERGED | `git merge-base --is-ancestor 6c84c7d5d origin/master` | **rc=0** |
  | what that object is | `git log -1 --format='%H %ci %s' 6c84c7d5d` | `6c84c7d5d…` `2026-08-27 21:32:31 +0200` `model: add Qwen3.8-Flash-Next (qwen4exp) (#27742)` |
  | a RELEASED tag has it | `git tag --contains 6c84c7d5d \| head -3` | `b10660`, `b10661`, `b10662` |
  | the PINNED object is no longer on master | `git merge-base --is-ancestor 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3 origin/master` | **rc=1** — the PR branch was squashed away |

  A follow-up, `6fe749801` "model: qwen4exp: reduce number of graph splits
  (#27880)", lands after it. The consequence for THIS record is only that the
  oracle file's framing is wrong; the pin itself is a recorded object with
  recorded build and run evidence, and advancing it re-measures both. That is a
  separate gated operation and W5p did not touch the pin, the `oracle-pin` block,
  or any measurement taken against it. W5p read `6fe749801` for the tensor-op
  declarations it cites and says so at every citation, which is a READ of a
  merged upstream and not a change of denominator.

- **THE FIXTURE IS FLOAT-ONLY WHERE THE ARTIFACT IS QUANTIZED, AND THAT IS A
  GENERAL GAP, NOT ONE OP'S. ISSUE OWED.** `qwen4_exp_gguf_fixture.h` writes
  every tensor at ggml type `0` EXCEPT ONE: `per_layer_token_embd.weight` is
  emitted at type `8` (Q8_0) on line 363, because the n-gram gather is the one
  path this row already knew had to be exercised quantized. The published UD-IQ1_S file uses NINE encodings
  (F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ1_S, IQ4_NL, BF16) over 1224 tensors.
  The refusal above is the first place that difference reached a production path;
  it is unlikely to be the last, because every `vt::` op this loop composes has a
  dtype contract that the fixture exercises on exactly one side. A fixture arm
  that emits the real encodings — or a gate driven from the published header
  manifest already committed at `tests/vllm/models/qwen4_exp_gguf_manifest.inc` —
  would convert a class of latent refusals into failing tests.

- **THE 74-MINUTE LOAD WAS CIFS, AND STAGING TO LOCAL DISK MAKES IT 60 SECONDS.**
  W5n set `MODEL=/workspace/q4exp-bench/UD-IQ1_S/...`, and `/workspace` is
  `//192.168.68.102/Data[/rc]` over SMB 3.1.1 while the worker's own overlay had
  503 GB free and went unused. Measured on **thor**, the box W5n actually ran on
  (its header comment says dgx; the run recorded 14 cores, `Mem: 122`, tegra,
  worker `rc-worker-n8smh`), inside one lease, same binary, same artifact:

  | | rate | how |
  |---|---|---|
  | `/workspace` sequential read | **20.9 MiB/s** | 2.0 GiB in 98.28 s, cold |
  | worker-local `/tmp` sequential read | **953 MiB/s** | 2.0 GiB in 2.11 s, cold |
  | staging copy, 67.564 GiB CIFS -> local | **23.7 MiB/s** | 2916 s |

  Both `dd` arms were cold: `posix_fadvise(DONTNEED)` then `mincore` VERIFIED
  `cached kB = 0` before each. Local storage is **45.6x** the share.

  **Load, same binary, only the filesystem changed:** 4446 s from CIFS ->
  **60 s** from local disk, a **74x** difference. The split is
  `copy 2916 s + load 60 s`, never one number: paying the copy once still beats
  the direct CIFS load (2976 s vs 4446 s) and every RELOAD after it is 60 s.

  **CIFS explains essentially all of it.** Our loader's own host work is bounded
  by that 60 s, i.e. **1.3%** of the 74 minutes. Taking the share's own measured
  ceiling, the 64.748 GiB the prefault pages would cost ~3172 s at 20.9 MiB/s, so
  ~71% of W5n's wall is the filesystem at its BEST sequential rate; the residual
  ~1270 s is the gap between our per-span access and one bulk `dd` stream, plus
  whatever contention differed between the two windows.

  **Where the local 60 s goes** (`VT_LOAD_STATS=1`, which this change made work on
  the GGUF path at all): `mmap+header 0.052 s`, `weights 49.482 s`, of which
  `prefault paged_in 64.748 GiB in 30.959 s (2141.6 MiB/s)` over 290 spans. With
  `VT_GGUF_PREFAULT=0` the whole load is **15 s** (`weights 13.120 s`, spans 0).
  So the prefault is 45 s of the 60 s locally -- and on CIFS that same eager,
  synchronous residency is what turns a slow mount into 74 minutes. `cpu_frac`
  0.61 locally: the prefault is page-fault bound, not disk bound (it beats the
  953 MiB/s single-stream `dd` because `madvise(WILLNEED)` readahead is wider).

  **The three suspects the brief named are all excluded, by routing not by
  timing.** Of 1224 tensors, 67.22 GiB is quantized and 0.33 GiB is F32/BF16;
  every quantized type present has a `vec_dot`, so all of it is BORROWED.
  `expand_nk` orients a bf16 expansion 99.5% of this file never takes; the i8mm
  repack is Q8_0-only (`QuantRepackEligible`) and Q8_0 is 0.74 GiB, ~1%; and seek
  thrash is unavailable because both shards have `table_order == file_order` with
  every `blk.N` one contiguous run and the mass in 49 IQ4_NL tensors of ~1 GiB.

  **OWED, not filed because GitHub writes are 403 from this host:**
  1. Stage model weights to worker-local storage before loading. Every GGUF job
     under `/workspace/q4exp-*` reads them in place; `.agents/environment.md` says
     to BUILD in `/tmp` and does not say to stage WEIGHTS.
  2. A job-template defect: `exec > >(tee ...)` plus a bare `wait` in an EXIT trap
     hangs the script after it finishes, holding the fleet device. Attempt 1 of
     this job held thor that way, and W5n has the same shape -- its log prints
     `=== DONE ===` and never prints `cleanup done`.
  3. No same-box llama.cpp control exists. On thor the `llama-cpp-qwen4exp` pin
     ABORTS before loading any weights: `GGML_ASSERT(obj_new) failed`
     (`ggml.c:1804`) from `ggml_transpose` <- `build_delta_net_chunking` <-
     `llama_model_qwen4exp::graph::build_layer_attn_linear`, during
     `graph_reserve`. The 16m27s dgx figure stays non-comparable, and it was
     always doubly so: different device (`n_threads = 20`) and an ASYNC
     `posix_madvise(WILLNEED)` prefetch against our synchronous residency.
  4. Run-to-run spread on the local load is wide -- two nominally identical local
     arms read 60 s and 45 s -- so 60 s is a scale, not a precise figure. The
     `local-warm` arm did NOT achieve a warm cache (its own precondition print
     shows ~3.2 GiB of 67.5 GiB resident), so the host-work floor comes from the
     prefault-off arm (15 s), not from it.

- **W5n's OWN ISSUE IS OWED.** GitHub writes are `403` from this host, so nothing
  could be filed. W5n rides under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031).

- **W5L's ISSUE IS OWED, and so is the ISSUE FOR THE WAVE IT PROPOSES.** GitHub
  writes are `403` from this host (account suspended), so nothing could be filed
  and no row was appended to `.agents/issue-index.md`; an index row pointing at
  an issue that does not exist is worse than an absent one. W5L rides under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031). Two issues are owed:
  one for W5L itself, and one for the RAGGED MULTI-REQUEST BATCH described in the
  next entry, which the `serves_one_sequence_per_step` clamp names in the message
  it prints to every operator.

- **`num_reqs > 1` IS A SEPARATE WAVE, AND HERE IS THE MEASURED SPLIT.** W5L was
  asked to address it if it blocked serving. It does not block serving, it blocks
  BATCHED serving, and the two have different costs. What W5L did instead is make
  the refusal SURVIVABLE: the forward's refusal is thrown inside the EngineCore
  busy loop, which treats a throw as fatal, so before this wave a server at
  `--max-num-seqs 4` answered three overlapping `/v1/completions` calls with three
  500s and never served again — at the DEFAULT `max_num_seqs` of 128 that was the
  out-of-the-box behaviour. `ModelFactory::serves_one_sequence_per_step` plus the
  clamp in `LoadedEngine::ResolveMaxNumSeqs` turn that into serialized service.
  The refusal in the forward is UNCHANGED and still fires for anyone who builds a
  batched step by hand. What the batching wave owes, measured on this tree rather
  than estimated:
  - `RunQwen4ExpQsaBlockPaged` takes `block_table` as i32 `[1, max_pages]` and
    asserts it twice (`qwen4_exp_qsa_block.cpp:432` and `:457`, both saying "this
    block serves ONE sequence per call"). A ragged batch needs a `[num_reqs,
    max_pages]` table and `query_start_loc` threaded through the indexer, the
    top-k selection and the gather consumer.
  - The PLE layer's `state_row` is a SCALAR
    (`caches.ple[i].state_row`, read from
    `gdn_meta.non_spec_state_indices_tensor[0]`), so the conv ring and the n-gram
    history are addressed one sequence at a time.
  - The mRoPE position table and the hyper-connection stream are built for one
    contiguous query range.
  None of that is a hook widening; it is three block seams. It is W5m, and the
  clamp's stderr line names this spec so the operator who hits the ceiling finds
  the entry.

- **PAID BY W5n, AND THE ROW IT PAYS SAYS THE ARM DOES NOT DECODE.** This entry
  read "THE `docs/USAGE.md` WEIGHTS ROW IS STILL OWED, AND W5L IS NOT THE CHANGE
  THAT PAYS IT", on the correct ground that every byte W5L served came from
  `tests/support/qwen4_exp_gguf_fixture.h`, a synthetic file whose weights are a
  deterministic ramp. **W5n read the published bytes.** `rc` job `0f188dd1` on
  `thor:gpu0`, 2026-08-30, drove `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S
  (67.564 GiB, 3 shards, 1224 tensors) through `examples/server` on
  `--device cpu`. The registry row now states a size, three sha256 values, a
  revision and an arm that WAS measured — and what it records is that the arm
  **LOADS and LISTENS and produces ZERO TOKENS**: 4446 s to `/health`, 69.206 GiB
  peak RSS, then `POST /v1/completions` returns 500. See
  [the evidence](../../docs/bench-evidence/qwen4exp-released-checkpoint-serve-20260830.md)
  and the entry below for the refusal. A row that claimed a served completion
  would have been the row nobody measured; this one says what happened.

- **THE FIXTURE'S TOKENIZER IS 16 SINGLE-CHARACTER TOKENS, SO `/v1/chat/completions`
  CANNOT BE EXERCISED ON IT.** `FixtureOpts::with_tokenizer` emits a byte-level
  BPE over `'a'..'p'` because the model's vocabulary is 16 wide, and the chat
  endpoint's fallback role-join prompt contains the literal word `user`, whose
  `u` is not in that alphabet — measured: `500 tokenizer: symbol "u" not in vocab
  (incomplete byte-level alphabet?)`. That is a property of the FIXTURE and not of
  the chat path, and it is recorded rather than worked around: widening the vocab
  would move every value the oracle golden in `test_qwen4_exp_layer_loop.cpp`
  measures. `/v1/completions` over in-alphabet prompts is what W5L gated and what
  it claims.

- **(SUPERSEDED by W6-CUDA.)** This entry read "NO CUDA ARM WAS BUILT OR RUN.
  There is no CUDA kernel for any `qwen4_exp` op, so the served path is
  `--device cpu` and a device run is not merely unmeasured, it is unavailable."
  That was true of W5L and is no longer true of the tree: three of the six ops
  now have CUDA arms, compiled and run on `sm_110`. It is kept, struck, because
  the wave that says "it serves" is the wave a reader will quote, and a reader
  arriving here needs to be sent forward rather than told something false. **The
  served path is STILL `--device cpu`**, for a different reason that has not
  moved: `ModelRegistry::Forward` is all-or-nothing and the remaining four ops
  plus `vt::RmsNormGroup` plus the block-decoding n-gram gather have no CUDA arm,
  so a `qwen4_exp` step still cannot reach a CUDA queue.

- **W5j's ISSUE IS OWED.** GitHub writes are `403` from this host (account
  suspended), so nothing could be filed and no row was appended to
  `.agents/issue-index.md`; an index row pointing at an issue that does not exist
  is worse than an absent one. The change rides under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) and
  [#2353](https://github.com/mudler/vllm.cpp/issues/2353).

- **(RESOLVED by W5k, repaired in flow by W5L.)** This entry read "THE SECOND
  STEP IS NOW BLOCKED ON ONE ENTRY AND ONE ENTRY ONLY" and pointed a reader
  scanning for "what stops a decode" at the PLE dtype/residency entry below.
  NOTHING STOPS A DECODE: W5k settled both against the running lane pin and the
  entry it points at is itself marked RESOLVED, so this one was a live pointer to
  a dead fact — exactly the drift it was written to avoid. It is kept, struck, and
  the original text follows for provenance. **W5j**
  removed every other reason `past_len != 0` refuses — the channel reaches the
  hook, all five published caches resolve by name, and group 2's pages persist —
  so what remains is the PLE conv dtype and the n-gram history's residency, which
  this file already tracks under **THE PUBLISHED PLE CONV STATE IS bf16 AND
  `RunQwen4ExpPleBlock` REQUIRES f32** below. That entry carries W5j's
  re-measurement and its two anchor corrections; this one exists so a reader
  scanning for "what stops a decode" finds one answer and not two. Not
  duplicated on purpose: a second copy of a live claim is the drift lock this
  section keeps filing against itself.

- **THE BY-NAME PATH IS CPU-ONLY, FOR THE SAME REASON THE PAGED ARM ALWAYS WAS.**
  `IndexerRows` refuses a device-resident block table by name, and the hook hands
  it group 2's gathered row through a `dense_attn::DBuf`, which is CPU-resident on
  a CPU queue and device-resident on a CUDA one. So a CUDA step reaches the by-name
  resolution and then stops inside the block. This is the same owed CUDA arm the
  QSA ops already carry and it is not new debt, but W5j is the wave that makes it
  reachable, so it is named here. A cheaper fix than a device translation exists —
  the table is read on the host and nowhere else, so it could be a host tensor over
  the runner's own vector rather than a `DBuf` — and it is deliberately NOT taken
  here, because it would change the POSITIONAL arm's shape too and this wave's
  gate does not cover that.

- **THE SHARED GGUF FIXTURE CANNOT GATE CACHE CONTENT, MEASURED.** Across two
  different prompts through the by-name path, **0 of 128** indexer-cache words and
  **0 of 192** paged-K/V words moved, while the logits moved by 31.84. It is a
  dynamic-range property and not a defect: the four-layer ramp puts the layer-3
  activations near 2^18, where one bf16 ULP is about 1024, and 31.84 of 95090 is
  0.03% — an order of magnitude under one ULP. The paged K/V, whose store is
  gated by every other model in the tree, is invariant by the same count, which is
  what separates "the fixture saturates" from "the indexer writes one row T
  times". So the ROW SET is what gates the paging in `test_qwen4_exp_layer_loop`
  and the CONTENT is gated at the block instead. Owed: a fixture rescaled so cache
  content is prompt-separable, which would let this suite gate both.


- **W5h's ISSUE IS OWED.** GitHub writes are `403` from this host (account
  suspended), so nothing could be filed. The change rides under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) and no row was
  appended to `.agents/issue-index.md`, because an index row pointing at an
  issue that does not exist is worse than an absent one.

- **`.agents/issue-index.md` ROW [#1978](https://github.com/mudler/vllm.cpp/issues/1978)
  NOW CARRIES A FALSIFIED SENTENCE AND CANNOT BE REPAIRED.** Its survey reads
  "one stored state per 4 tokens via `MLAAttentionSpec(tokens_per_state=
  compress_ratio)`" among the nine DeepSeek-V4 structural matches. That is true
  of DeepSeek-V4 and NOT of `qwen4_exp`, whose indexer caches the RAW per-token
  key (`modeling_qwen4_exp.py:655`, `cache_utils.py:346`) — the ninth "match"
  is the one that does not hold, and believing it is what produced the 4x-short
  group W5h fixes. The index is APPEND-ONLY and carries `merge=union`, so the
  row is left byte-for-byte alone and the correction lives here, which is the
  arrangement AGENTS.md prescribes for a keyed append-only record.

- **CLOSED by W5i: THE INDEXER SIDE CACHE IS ADDRESSABLE THROUGH THE PAGED
  ALLOCATION.** This entry read "now big enough and still not readable".
  `Qwen4ExpQsaPagedCaches` now carries the side cache as the runner's own fused
  MLA page `[num_pages, block_size, indexer_head_dim]` plus GROUP 2'S OWN block
  table, the store scatters through `vt::IndexCopy` and the read gathers through
  `vt::IndexSelect`, and `Qwen4ExpQsaIndex`'s contiguous `[rows, D]` contract is
  unchanged because the gather is what hands it one. **NO NEW OP AND NO OP
  EXTENSION**: the entry's own claim that `vt::IndexSelect` and `vt::IndexCopy`
  suffice was re-verified against their contracts before anything was written,
  and it holds — `IndexCopy` wants a contiguous `[N, D...]` destination, which is
  exactly what an MLA group's pages are when flattened. The golden uses a
  NON-IDENTITY permutation `{2, 6, 1}` against a logical `{0, 1, 2}`, DIFFERENT
  from the K/V arm's `{5, 3, 7}` so that resolving one group through the other's
  map is visible, with a PARTIAL final page. W5i's own issue is OWED.

  **WHAT THE GOLDEN MEASURED THAT THE ENTRY DID NOT ANTICIPATE, and it changes
  what a future wave here must assert.** The store and the read share one
  translation, so a translation that is wrong THE SAME WAY on both sides writes
  and reads the same wrong rows and RETURNS THE RIGHT ANSWER. Measured, not
  feared: against a body that dropped the permutation entirely, the paged-vs-
  contiguous comparison read `differing 0 of 1472` and the oracle bound read
  `0.00558036`, both green, while 9 of 23 rows were in the wrong physical page.
  A value comparison at prefill is therefore NOT a gate on this. What convicts is
  the STRUCTURAL assertion — the exact set of physical rows written, against the
  set the block table names, with every other row still the NaN it was
  constructed with — and a decode step whose prefix the TEST places at the rows
  it computes itself. Both are in `test_qwen4_exp_qsa_block.cpp`'s W5i case.

  Three narrower things replace this entry, and each is named where it belongs
  below: the ENGINE's group-2 buffer still does not reach the block (W5j);
  the gather costs one extra pass over the visible prefix per layer per step;
  and the page translation is a HOST read of the block table, so a device-
  resident table is refused by name and the CUDA arm owes it a device-side home.

- **CLOSED by W5j — THE FORWARD RESOLVES ITS CACHES BY NAME AND THE ENGINE GUARD
  IS A PER-ARCHITECTURE CAPABILITY.** All three bullets below were measured
  before the wave and all three are done: the hook resolves through
  `MultiKvCacheIndex::Resolve`, the three published names are built by ONE
  file-local builder both `MakeQwen4ExpKVCache` and the forward call, and
  `ModelFactory::consumes_multi_kv` is the declared bit. M4 drives the guard red
  by clearing it, against its own message and not a bare `CHECK_THROWS`. The
  bullets are kept as written because they are the measurement that scoped the
  wave. What is NOT closed is the second step; see the PLE entry below. The
  ORIGINAL text, still accurate as a statement of the gap W5j found:
  - `ForwardQwen4ExpForConditionalGeneration` asserts
    `input.attn_kv.size() == n_qsa` (`qwen4_exp_registry.cpp`, the "paged K/V
    caches for ... qwen_sparse_attention layers" refusal). The runner allocates
    one paged buffer per (ATTENTION GROUP x LAYER) on the multi-cache path
    (`src/vllm/v1/worker/gpu/runner.cpp:1300-1339`, the loop over
    `attn_group_ids_`), and this model publishes TWO attention groups, so
    `attn_kv` arrives at `2 * n_qsa`. Lifting the engine guard without fixing
    this yields a different refusal, not a token.
  - The by-name channel now covers recurrent members (`ENG-MULTIKV-BYNAME`), so
    `MultiKvCacheIndex::Resolve` can address all five of this model's published
    cache kinds. The hook must use it, and it must build its layer names through
    ONE builder shared with `MakeQwen4ExpKVCache` — two derivations of one name
    set is the shape that can disagree.
  - `ModelRegistry::Forward`'s `if (input.multi_kv != nullptr)` refusal must
    become a bit the MODEL declares, which the guard's own comment already
    names as the intended polarity ("the polarity `ModelFactory`'s existing
    `stage_on_load` and offload bits already use"). It must still refuse an
    architecture that declares nothing, and the mutation that proves it is
    "declare the bit for a shape nothing consumes and check it STILL refuses" —
    against the specific message, never a bare `CHECK_THROWS` an unrelated
    exception can satisfy.
  ISSUE OWED.

- **THE PUBLISHED PLE CONV STATE IS bf16 AND `RunQwen4ExpPleBlock` REQUIRES
  f32.** `MakeQwen4ExpKVCache` pushes the PLE conv state at
  `conv_dtype = vt::DType::kBF16`, and the block refuses anything else:
  `VT_CHECK(conv_state.dtype == DType::kF32 && conv_state.IsContiguous(), ...)`
  (`qwen4_exp_ple_block.cpp:320-321`), then reads it through
  `conv_state.Ptr<float>()` (`:354`). The n-gram history is `kI64` on both
  sides and agrees, but it is read through a HOST pointer (`:386-389`), so the
  runner's allocation is only usable while the device is CPU. Which side is
  wrong is a design call this wave did not have the oracle to settle — upstream
  stores whatever dtype the conv input carries (`update_conv_state`) — so it is
  RECORDED rather than changed, because publishing a dtype on a guess is the
  same class of error W5h just removed. It blocks multi-step decode either way.
  ISSUE OWED.

  **W5k SETTLED THIS AGAINST THE ORACLE AND BOTH SIDES NOW AGREE. RESOLVED.**
  The entry above and the W5j note below it are kept for provenance; the state
  they describe is gone. W5j STOPPED for the right reason but looked in the wrong
  place: it read a local `transformers` checkout at `7d06b1a5` and an installed
  wheel at 5.3.0, when the lane pin is a RELEASE. W5k installed 5.16.0 into a
  virtual environment, verified `modeling_qwen4_exp.py` sha256
  `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459` against the
  pin, and confirmed the environment by REGENERATING
  `tests/vllm/models/qwen4_exp_forward_goldens.inc` byte-identically
  (sha256 `d968a142…05d77`, unchanged). Then it read the running model:

  - **THE CONV RING CARRIES THE MODEL DTYPE, so the PUBLISHER was right and the
    BLOCK was wrong.** `cache_utils.py:1019-1023` allocates each slot as
    `torch.zeros(..., dtype=conv_states.dtype, device=conv_states.device)` — PER
    SLOT, from the tensor that first reaches it — and the tensor reaching this one
    is `hidden_states` (`modeling_qwen4_exp.py:1157-1159`). OBSERVED, not
    inferred: the same fixture at `dtype=torch.bfloat16` reports
    `conv_states[1] dtype=torch.bfloat16`, and at `float32` reports `float32`. It
    never widens. So `MakeQwen4ExpKVCache`'s bf16 was upstream's answer and the
    f32 requirement was the "dtype too wide" AGENTS.md names. The block now
    requires the ring to EQUAL the stream dtype, which is upstream's own
    construction rather than a widened admission.
  - **THE N-GRAM HISTORY IS DEVICE-RESIDENT, so the PUBLISHER was right again.**
    `:1070` takes `input_ids.long()` and `:1089-1091` hands exactly that to
    `update_conv_state(..., state_idx=2)`, so the slot's device is
    `input_ids.device` — the compute device. The old refusal stated a true fact
    about THIS TREE (the splitmix64 hash is a host int64 computation) and turned
    it into a requirement on the CACHE, which is the wrong object. The block now
    accepts either residency and stages the row — `ngram_size - 1` int64s, 16
    bytes at the released config — around its host hash.
  - **THE EOS SEED ON THE FIRST STEP IS CONFIRMED.** `:1073-1076`: when
    `has_previous_state(layer_idx, state_idx=2)` is false, `previous_context =
    input_ids.new_full((B, context_len), self.eos_token_id)`. This tree's
    `past_len == 0` / `prefill_has_initial_state == 0` branch mirrors it exactly.

  A **DEVICE splitmix64** is still worth having and is now an OPTIMIZATION rather
  than a blocker: it would remove the two 16-byte copies per PLE layer per step.
  ISSUE OWED.

  **W5j's RE-MEASUREMENT, kept for provenance (superseded by W5k above).** The anchors have moved: the host read of the n-gram history is
  `:351` and not `:386-389`, and the CPU-residency refusal is its own `VT_CHECK`
  at `:327`. And the reason for `past_len != 0` is now THIS entry alone — before
  W5j the indexer side cache and these two states all had "nowhere to persist",
  and since W5j the indexer side cache lives in the engine's group-2 pages and
  these two are PUBLISHED in the recurrent group and merely unreadable by the
  block. The oracle is still unavailable HERE and that was checked rather than
  assumed: the local `transformers` checkout is at `7d06b1a5` with no
  `models/qwen4_exp/` directory and the installed wheel is 5.3.0, so W5j STOPPED
  rather than guessed. The tree's own convention argues one way and is not the
  oracle: `nemotron_h_device.cpp:1313-1314` accepts bf16, f16 or f32 for a conv
  state and `qwen3_5.cpp:9621` hands one as bf16, so the block requiring f32 is
  this model's exception and AGENTS.md "Inherit vLLM defaults" makes an f32 model
  buffer the annotated-exception direction. **Until this is settled no second
  step decodes and the server can serve exactly one forward per sequence.**

- **A SECOND STEP DECODES; SERVING IS STILL NOT CLAIMED.** W5k removed the
  `past_len == 0` refusal and a decode at `past_len = 6` now returns a token
  through `ModelRegistry::Forward` over the engine's own persistent caches
  (`test_qwen4_exp_layer_loop.cpp`, "a SECOND step decodes on the engine's own
  persistent caches"). The clause this replaces said the second step was "blocked
  on the PLE conv dtype and the n-gram history's residency"; both are settled
  above. What remains before an `examples/server` end-to-end or a
  `docs/USAGE.md` weights row is owed (**the weights row itself is PAID by W5n**;
  what follows is the rest of that list, and W5n adds a new first item to it —
  the `qwen4_exp_gated_residual` quantized-operand refusal, below):

  - `num_reqs > 1` is still refused. `RunQwen4ExpQsaBlockPaged` takes a
    `block_table` of i32 `[1, max_pages]`, so a ragged multi-request batch needs
    `query_start_loc` plumbing no block on this row carries. ISSUE OWED.
  - The POSITIONAL arm still serves one shot, and that is now a statement about
    that arm rather than about the model: nothing publishes the PLE states there,
    so the hook allocates them per call and a per-call buffer is zeroed on entry.
    Refused on the same predicate that routes.
  - `GPUModelRunner` has not been driven end to end; the two steps above are
    assembled the way the runner assembles one, not BY the runner. ISSUE OWED.
  - The FIXTURE still cannot gate cache CONTENT (W5j measured 0 of 128 indexer
    words and 0 of 192 paged K/V words moving while logits moved 31.84; layer-3
    activations sit near 2^18 where one bf16 ULP is ~1024). W5k gates the
    cross-step path on the n-gram history's INTEGER token ids, which cannot
    saturate, and asserts no cache value. A RESCALED fixture stays owed.

  So no `docs/` surface changes in this wave.

- **CLOSED BY W5g, AND ITS DIAGNOSIS WAS HALF RIGHT.** The entry below is kept
  because its measurement is what bought the fix, and because BOTH repairs it
  proposed would have left the real defect standing: the layout, not only the
  fixture, took the wrong source as authority. See
  `## The PLE layout's two sources, and which one is the authority (W5g, #2031)`
  above. The issue is still OWED — GitHub writes are `403` from this host, so
  nothing could be filed for W5g either.

- **THE SHARED `qwen4exp` GGUF FIXTURE IS INTERNALLY INCONSISTENT, AND NOTHING
  COULD SEE IT UNTIL A FORWARD RAN THE PLE LAYER ON IT (found by W5f).**
  `tests/support/qwen4_exp_gguf_fixture.h` states
  `qwen4exp.ple.head_vocab_sizes = {23, 29}` and its own comment says those are
  "what the HF derivation would produce from `ngram_vocab_size_base = 20`". The
  GGUF CONTAINER HAS NO SUCH KEY: `Qwen4ExpHfConfigFromGguf` reads
  `ple.ngram_size`, `ple.heads_per_ngram`, `ple.conv_kernel`,
  `ple.head_offsets`, `ple.head_vocab_sizes`, `ple.layer_multipliers`,
  `ple.layers` and `ple.eos_token_id`, and no base — because
  ggml-org/llama.cpp#27742 writes the RESOLVED sizes instead. So the parsed
  config carries upstream's DEFAULT of 20,000,000, W5e-2's `Qwen4ExpPleLayout`
  derives 20,000,003 for head 0, and its cross-check refuses by name.
  **On a REAL file the two agree** — the released config's base IS 20,000,000 —
  so this is a fixture defect and not a port one, and it cannot be fixed by
  editing the stated sizes: a table addressed from base 20,000,000 needs forty
  million rows. The two candidate repairs are (a) teach the GGUF config builder
  an OPTIONAL `qwen4exp.ple.ngram_vocab_size_base` and have the fixture state
  20, which is inert for every real file but invents a container key the
  container oracle does not write, and (b) narrow W5e-2's cross-check so it
  compares stated sizes against a base the SOURCE stated rather than against a
  config DEFAULT, which is a checker-semantics change and needs its own spec and
  red-before. Both are larger than W5f and neither should be smuggled into it.
  **NO ISSUE NUMBER, and that is an outage and not an omission:** GitHub write
  access was suspended (HTTP 403) for the whole of this wave, so no issue could
  be filed and no `.agents/issue-index.md` row could name one. Owned by row
  `MODEL-MM-QWEN4-EXP` under
  [#2336](https://github.com/mudler/vllm.cpp/issues/2336) until it has its own.
  W5f's reachability case asserts the refusal BY NAME rather than working around
  it, so the day this is fixed the case fails loudly and says so.
- **`check-env-doc`'s RED IS THE BASE'S, AND THIS BRANCH NOW CARRIES `main`'s OWN
  REPAIR BYTE-FOR-BYTE RATHER THAN A SECOND DESCRIPTION OF THE SAME KNOB.**
  `VT_QWEN35_STAGE_RESERVE_BYTES` is read at
  `src/vllm/model_executor/models/qwen3_5_weights.cpp:213` and is documented
  nowhere at this branch's base `f060a81d6`, so `scripts/check-env-doc.py` and
  `tests/scripts/test_check_env_doc.py::test_shipped_tree_is_fully_covered` are
  both RED at the base. Measured with the checker's own pure functions against
  each revision's blobs: `f060a81d6` -> `['VT_QWEN35_STAGE_RESERVE_BYTES']`,
  W5f's first head -> `[]`, `c31b2496e` -> `[]`. W5f's first head closed it with
  a row that was **materially wrong in two ways**, which fresh review measured:
  it described `StagingFitsModel` as "stage only if the model still fits with
  `reserve` bytes left over" when the predicate is
  `2 * model_weight_bytes + reserve_bytes <= device_total_bytes`
  (`qwen3_5_weights.cpp:204`) — dropping the `2 *` admits exactly the
  double-booking [#1299](https://github.com/mudler/vllm.cpp/issues/1299) exists
  to prevent — and it KEPT the `VT_QWEN35_STAGE_MIN_FREE_FRAC` row asserting that
  "the fraction knob above is retained for the arm that still asks per weight",
  when no such arm exists: the only `getenv` in that file is
  `VT_QWEN35_STAGE_RESERVE_BYTES`, and `MIN_FREE_FRAC` survives only in a stale
  comment at `include/vllm/model_executor/models/qwen3_5_weights.h:1340`. The
  gate cannot see either defect, because it only demands that SOME row mention
  the name.
  **`main` had already fixed it properly at
  [`c31b2496e`](https://github.com/mudler/vllm.cpp/commit/c31b2496e)
  ([#2359](https://github.com/mudler/vllm.cpp/issues/2359)), whose title is
  literally "and drop the knob it replaced": it REPLACES the `MIN_FREE_FRAC` row
  with the `RESERVE_BYTES` one rather than adding beside it.** That commit is not
  an ancestor of this branch, because this branch is stacked on W5e-2's
  `f060a81d6` rather than on `origin/main` (GitHub write access was 403 for the
  whole wave). Fresh review asked for the wrong hunk to be dropped on the
  ground that "`check-env-doc` is green either way"; that premise is FALSE, and
  removing the hunk was measured to turn both gates red. So the substance of the
  finding is applied and the mechanism is not: this branch takes `c31b2496e`'s
  row VERBATIM, and `docs/ENVIRONMENT.md` is now byte-identical to
  `c31b2496e:docs/ENVIRONMENT.md`. The rebase onto `main` therefore resolves this
  file to the same bytes from either side and cannot land a competing
  description. The stale `MIN_FREE_FRAC` comment in `qwen3_5_weights.h` is NOT
  fixed here and is not this row's: it belongs to `PERF-QWEN35-STAGE-WEIGHTS`
  with [#2357](https://github.com/mudler/vllm.cpp/issues/2357).

- **W5f's REACHABILITY IS A PREFILL, NOT A DECODE, and the distinction is the
  whole of this row's honesty about itself.** `ModelRegistry::Forward` reaches
  `Qwen4ExpTextModelForward` on a model `ModelRegistry::Load` produced from a
  real `qwen4exp` GGUF — the first production forward this architecture has ever
  had — and the hook refuses `past_len != 0` and `num_reqs != 1` by name. What
  is owed was described here as "the engine's, in three named pieces, none of
  which belongs to this row". **TWO OF THE THREE ARE GONE AND THE THIRD WAS
  ALREADY WRONG WHEN IT WAS WRITTEN.** W5j closed
  `ModelRegistry::Forward`'s `multi_kv` refusal for this architecture, by the
  per-architecture capability bit the guard's own comment had named as the
  intended polarity, and it closed it in THIS row rather than in
  KV-DSV4-MULTICACHE W5 ([#1925](https://github.com/mudler/vllm.cpp/issues/1925),
  [#2068](https://github.com/mudler/vllm.cpp/issues/2068)), whose scope is
  DeepSeek-V4's model half. The second piece — "a channel that can address
  recurrent (`MambaSpec`) members, which the by-name index cannot" — was already
  false at W5i's head: `ENG-MULTIKV-BYNAME` had added `payload_kinds` /
  `payload_slots` and made all five of this model's caches resolvable, which is 5
  of 5 and not 2 of 5, and this entry never caught up. What REMAINS is the
  `query_start_loc` plumbing a ragged multi-request batch needs, which no block on
  this row carries. **THE REST OF THAT SENTENCE IS NOW WRONG AND IS REPAIRED IN
  FLOW BY W5L.** It continued "PLUS the second-step blocker above, which is this
  row's own and not the engine's. Until both land there is no token number, no
  speed number, no `examples/server` end to end and no `docs/USAGE.md` weights
  row." W5k landed the second step and W5L serves a `POST /v1/completions`
  through `examples/server` on CPU, so BOTH did not have to land: the ragged
  batch is still owed and serving does not wait on it, because
  `ModelFactory::serves_one_sequence_per_step` clamps the engine to one sequence
  rather than letting it build a step the forward refuses. What survives
  unchanged is the pair that never depended on batching: there is no token number
  and no speed number, because every byte served came from the synthetic fixture,
  and the `docs/USAGE.md` WEIGHTS row was owed by the wave that serves a
  published checkpoint. **W5n PAID IT** — see the entry above and
  [the evidence](../../docs/bench-evidence/qwen4exp-released-checkpoint-serve-20260830.md).
  It is paid as a LOAD, not as a decode: the released artifact loads and the
  server listens, and the forward then refuses it by name.
- **CLOSED by W5i as a STORE, and what remains is a REACH.** This entry read
  "the QSA indexer side cache's paged store is still owed". The store and the
  read are paged now, and the registry hook allocates the scratch IN THE ENGINE'S
  OWN PAGED SHAPE and addresses it through a table, so the code the engine's
  buffer will run is the code that runs today. **THE REACH IS CLOSED TOO, BY
  W5j.** On a step carrying `multi_kv` the hook hands the block group 2's OWN
  buffer, viewed as the fused MLA page, and `group_block_tables[<group 2>]` — the
  vector W5c-2 already gathers — through `BlockTableForGroup`. This entry's
  prediction held to the letter: the substitution was two lines in
  `ForwardQwen4ExpForConditionalGeneration` and nothing in the block, which is
  what paging it here bought. The scratch arm SURVIVES for a caller with no
  engine behind it, where the IDENTITY table is the correct map because there is
  no allocator and so no physical pages to permute; it is no longer the
  production path, so the permutation is exercised by the allocator's own pages
  as well as by the block's gate.
- **THE GATHER COSTS ONE EXTRA PASS OVER THE VISIBLE PREFIX, per QSA layer per
  step, and it is recorded rather than hidden.** `vt::IndexSelect` materialises
  `[kv_len, indexer_head_dim]` before `vt::Qwen4ExpQsaCompress` reads it. It is
  NOT an asymptotic change — the compressor already streams every visible row
  each step, rebuilding every pooled block key and caching none, exactly as
  upstream does (`modeling_qwen4_exp.py:679-682`) — but it is a real constant on
  a pass that is already O(kv_len). Folding the page resolution INTO that op, as
  W5d-3 did for `vt::Qwen4ExpQsaGatherAttention`'s K/V read, removes it. That was
  weighed and not taken: the compressor is a ONE-ARM op with no CUDA arm, so an
  address mode there is an unmeasurable extension, and no benchmark on this row
  can price it until a decode exists. **NOTHING HAS MEASURED THIS**, and the
  issue that owes the measurement is OWED with W5i's.
- **THE INDEXER PAGE TRANSLATION IS A HOST READ.** `IndexerRows` reads group 2's
  block table on the host to resolve a physical row, and REFUSES a table that is
  not CPU-resident by name rather than dereferencing a device pointer. That is
  `CheckRopeLayoutsAgree`'s rule in the same file, for the same reason, and it is
  owed the same thing: the QSA CUDA arm must give the translation a device-side
  home or argue it away. Recorded here so the CUDA wave inherits two items and
  not one.
- **THE MoE ADAPTER IS REBUILT PER LAYER PER STEP, and that is a SPEED ceiling
  W5f accepted rather than a wrong answer.** `Qwen4ExpMoeBlockWeights` runs
  inside the layer loop, which is the third risk #2336 §3 named: a per-step
  adapter copy loses `ResidentWeight::d_dev` and re-uploads the tower on a
  device arm. It is correct on every arm and free on CPU (the per-expert views
  are zero-copy borrows). Hoisting it to load time is owed with the CUDA arm,
  and no speed claim on this row is admissible before it.
- **`in_proj_ba` AND `in_proj_qkvz` STAY EMPTY ON THIS ARM, so
  `vt::GdnPackedDecode` never fires.** `Qwen4ExpGdnBlockWeights` fills the SPLIT
  fields, which is exact parity with qwen3_5's own GGUF path and the reason
  #2336 §3 listed it as a non-arithmetic risk rather than a defect. It is a
  performance ceiling this row inherits from the loader it shares, and it is
  owed with whatever wave gives the GGUF path a merged `in_proj_ba`.
- **THE LOOP IS GATED AT ONE GEOMETRY AND ONE DTYPE.** The golden runs four
  layers at `hidden_size` 8 in bf16 against an f32 oracle. It does NOT run: the
  released 48-layer geometry (no artifact fits a fleet device — the lane pin's
  own `gateable = no` reason), a quantized arm (`qwen4_exp_forward_goldens.inc`
  is bf16-from-f32 and nothing drives a keep-quant expert tower or n-gram table
  through the loop), a CUDA arm (every `vt::` op the PLE block is the first
  production caller of is registered on `kCPU` alone, and the n-gram id build is
  a host round trip by construction), or a masked prefill (`conv_mask` is
  `nullptr` on both cases). The k-quant obligation the previous entry recorded
  as "owned by W5f at the earliest, because nothing before the layer loop loads
  one" is therefore NOT discharged: W5f loads one, and its own gate does not
  drive the quantized arm through the loop.
  **AND THE RECORD HALF IS SATISFIED WHILE THE REFUSAL HALF IS NOT.** AGENTS.md
  ("A model port includes the quantized arms, not only bf16") asks for a refusal
  "with a message that names the missing part". The loop does not have one. Its
  dense projections — the GDN in/out projections, the QSA projections, the two
  PLE projections and the `lm_head` — carry no qwen4_exp-scoped dtype guard at
  all, so a k-quant weight arriving at one is refused by the shared `vt::`
  op-level dtype check, whose message names an op and a dtype and never names
  this architecture or its owed arm. The one exception is
  `qwen4_exp_moe.cpp:204`, which refuses a non-bf16 expert tower by seam name.
  Recording the arm as owed here therefore discharges the RECORD obligation and
  leaves the REFUSAL obligation open; it is owed with the arm itself, on the same
  wave, and is written down so the gap is not read as covered by this entry.

- **A REFUSAL THAT ENUMERATES PROSE GOES STALE SILENTLY, AND NOTHING PREVENTS THE
  FOURTH INSTANCE.** [#2288](https://github.com/mudler/vllm.cpp/issues/2288) is
  fixed by [#2265](https://github.com/mudler/vllm.cpp/pull/2265) for the two
  items that had gone false, but only those two: the DURABLE fix is not made
  here. `tests/vllm/models/test_qwen4_exp_scaffold.cpp:767` pins that five
  substrings are PRESENT, never that any of them is still TRUE, so a refusal
  listing finished work satisfies every assertion — it is a spelling gate, not a
  truth gate. This is the third instance on this row in one day (#2276 for the
  paged QSA consumer, #2254 for the opposite polarity, an understated refusal).
  A truth-linked check — each enumerated item naming a symbol whose absence the
  suite verifies — or a convention that the refusal enumerates ISSUE NUMBERS
  rather than prose would close it, and both are larger than this flow and belong
  to whoever owns the reachability convention. Recorded here so the residual has
  a named home rather than living only in the issue.

  **AND THE FOURTH INSTANCE THEN ARRIVED, EXACTLY HERE.** Merging `main` into
  W5d-4 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4) made the
  refusal's third enumerated item — "an adapter from the stacked [E, I, H]
  qwen4_exp MoE tensors onto MoeBlockWeights" — false, because that adapter is
  what W5d-4 adds; landing it unedited would have shipped a commit whose own
  product output denies the commit, which is the #2288 finding word for word.
  It is repaired in this flow and the count further down is restated with it.
  The heading above is left standing rather than softened, because it was right:
  nothing mechanical caught this one either, a reading did, and the durable fix
  is still owed.
- **W5e-2 (#2336) lands UNREACHED, by AGENTS.md "Nothing lands dead", and its
  reachability mutation is VACUOUS rather than passing.**
  `RunQwen4ExpPleBlock` (`src/vllm/model_executor/models/qwen4_exp_ple_block.h`
  and `.cpp`) is reached at this merge commit only by
  `tests/vllm/models/test_qwen4_exp_ple_block.cpp`.
  `grep -rln RunQwen4ExpPleBlock src include examples tests` returns the block's
  own header and body, the test, `tests/CMakeLists.txt`, and prose mentions
  inside the production refusal that names it — no call. There is therefore NO
  production call site to delete, so `.agents/reachability.md`'s mutation has
  nothing to remove and is recorded as VACUOUS, never as a pass. Wiring it is the
  LAYER LOOP, `Qwen4ExpTextModel::Forward`, owned by row `MODEL-MM-QWEN4-EXP`
  W5f under [#2336](https://github.com/mudler/vllm.cpp/issues/2336) and
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978).
  `ForwardQwen4ExpForConditionalGeneration` still refuses by name before any
  downcast, and that refusal was EDITED IN THIS FLOW because this wave falsified
  its "there is no RunQwen4ExpPleBlock … and it is the LAST one" clause; the
  emitted bytes were read back out of the running hook twice to prove the repair
  and to catch a second defect in it ("on main" where "in this tree" is the only
  checkable claim from a branch).

  **WHAT THIS WAVE DOES MAKE REACHABLE, one hop short of production, is three
  seams that had NO caller under `src/` at all**, and the count is read off the
  tree rather than claimed: `qwen4_exp::BuildNGramIds` (W2's splitmix64 hash, whose
  only caller was `PleForward` in its own translation unit), `vt::RmsNormGroup`
  (landed by W5d-1 FOR PLE's three grouped norms and never routed to) and
  `vt::Qwen4ExpPleGate` (W5e-1). `vt::RmsNormGroup` and `vt::Qwen4ExpPleGate` go
  from zero callers under `src/` to one; `BuildNGramIds` gains its FIRST CALLER
  OUTSIDE ITS OWN TRANSLATION UNIT and so has two, `PleForward` at
  `qwen4_exp_ple.cpp:381` and the block at `qwen4_exp_ple_block.cpp:276`. In each
  case the new caller is this block. That is a real change in the reachability graph and it is
  NOT a decode; the distinction is the whole of this entry.
- **The f32 SCORE ACCUMULATOR at model width is a CONDITIONING HAZARD OF THE
  MODEL, not a port defect, and it is bounded rather than gated to `kTol`.**
  W5e-1's `## Owed` recorded that `vt::BatchedMatmul` accumulates the `:1180` dot
  in f32 with a sequential-over-K loop and that a sign flip inside the clamp band
  follows. W5e-2 measured it at `hidden_size = 2560`, `hc_count = 4`: worst score
  disagreement against a double accumulator 1.06371e-4 in `g` units, 2 sign flips
  in 3 near-null pairs, worst sigmoid delta 1.48301e-3 — 148 times the suite's
  1e-5 — against 2.09894e-9 on the dense pairs. The gate is the analytic bound
  `0.5 * sqrt(max(dg/2, 1e-6))`, derived from the clamp's floor and sigmoid's
  1/4-Lipschitz constant, and it is a GLOBAL bound rather than a local one.

  **MIRRORING vLLM MEANS KEEPING THE f32 ACCUMULATION.** Upstream's `:1180` is an
  f32 reduction too (torch's `acc_type` for a float sum is float; its pairwise
  blocking is a different ORDER, not a wider accumulator) and upstream's `:1181`
  is genuinely discontinuous at the origin, so two legitimate summation orders
  differ near zero by construction. `vt::BatchedMatmul` offers no wider
  accumulator and adding one would be the divergence. **What is owed is not a
  fix but a CONSEQUENCE for whoever writes the first token gate on this
  architecture:** a near-null PLE gate cannot be held to 1e-5 at model width by
  ANY port, so a token-exact claim on this model must either avoid the band or
  argue it away.

  **A MASKED ROW IS HARMLESS BECAUSE OF THE MASK, NOT BECAUSE `value` IS ZERO,
  AND IT IS NOT THE ONLY CASE.** An earlier draft of this entry said both, and
  the fresh review falsified both. `value` is `vt::MatmulBT(embeddings,
  value_proj)` at `:1178` and is NOT zero on a masked row; what makes such a row
  harmless is that the mask at `:1185-1187` runs AFTER the gate and zeroes both
  the gated output and the conv input, so whatever the gate returned is
  DISCARDED. A NON-MASKED row does reach the band at model width: measured over
  2,000,000 random RMS-normalised pairs at `hidden_size = 2560`,
  `rms(dg) = 9.05e-7`, worst `dg = 9.72e-6`, `|g| < 1e-6` on 4 and `|g| < 1e-3`
  on 1636 — a sign-flip rate of order 1e-6 per score, which at `hc_count = 4` and
  8192 scores per 2048-token prefill is of order once per 1e2 prefills. Flip
  COUNTS at this size are Poisson with a mean near 2 (this draw 1, the review's
  independent draw 0), so the rate is the number and neither count is. The output
  deviation is bounded per hc block by `0.5 * sqrt(max(dg/2, 1e-6)) * |value|`:
  5.0e-4 * `|value|` at the typical `dg`, where the clamp floor dominates, and
  1.1e-3 * `|value|` at the worst `dg` measured. Owned by W5f and by whatever
  wave first claims a token number.
- **The quantized arm of the two PLE projections is REASONED ABOUT, NOT
  MEASURED.** `LoadMatmul` (`qwen4_exp_weights.cpp`) can route `ple_key.weight`
  and `ple_value.weight` to keep-quant blocks, and `vt::MatmulBT` auto-dispatches
  `kMatmulBTQuant` on a block-typed weight, so `RunQwen4ExpPleBlock` needs no arm
  of its own — but `qwen4_exp_ple_goldens.inc` is f32, the block's gate runs f32
  and bf16, and nothing here drives a quantized `key_proj`. GGUF k-quants are a
  standing requirement (AGENTS.md, `porting-a-model.md`), so this is owed as a
  gate and not as an implementation. **Owned by row `MODEL-MM-QWEN4-EXP` under
  [#2336](https://github.com/mudler/vllm.cpp/issues/2336), by the same wave that
  first runs this block against a `qwen4exp` GGUF — W5f at the earliest, because
  nothing before the layer loop loads one.** An unowned k-quant entry is how a
  standing requirement gets lost, which is why the owner is named here rather
  than left to the reader. The n-gram TABLE's keep-quant arm is in the same
  position here and is gated directly by W6a.
- **The DEVICE arm of the n-gram hash.** `qwen4_exp::BuildNGramIds` is host int64
  bit-mixing and `RunQwen4ExpPleBlock` therefore performs a host round trip per
  step: the ids are built on the host and uploaded for `vt::Embedding`. The
  block's header already names the batching seam W2 left for it — one kernel over
  `[B, T, ngram_heads]` — and `Qwen4ExpPleCaches::tokens` is refused by name
  unless it is CPU-resident precisely so that this constraint is visible rather
  than silently violated. Owed with the CUDA arm, and it is a SPEED item on the
  decode path, not a correctness one.
- **W5e-1 (#2336) lands UNREACHED, by AGENTS.md "Nothing lands dead", and its
  reachability mutation is VACUOUS rather than passing.** `vt::Qwen4ExpPleGate`
  (`include/vt/ops.h`, dispatcher `src/vt/ops.cpp`, CPU kernel
  `Qwen4ExpPleGateKernel` in `src/vt/cpu/cpu_qwen4_exp_ple.cpp`, name in
  `src/vt/op_provider.cpp`) is reached at this merge commit only by
  `tests/vllm/models/test_qwen4_exp_ple_gate.cpp`. A tree-wide grep over
  `src/ include/ examples/ tools/ benchmarks/` returns only the op's OWN
  declaration, dispatcher, kernel, registration and name entry, plus two prose
  mentions inside the production refusal that names it as unreached. There is
  therefore NO production call site to delete, so
  `.agents/reachability.md`'s mutation has nothing to remove — it is recorded as
  vacuous, never as a pass, which is the distinction that guide's step 5 draws.
  Wiring it is the PLE BLOCK, `RunQwen4ExpPleBlock`, owned by row
  `MODEL-MM-QWEN4-EXP` W5e-2 under
  [#2336](https://github.com/mudler/vllm.cpp/issues/2336) and
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978).
  `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  before any downcast, and that refusal was EDITED IN THIS FLOW because its "the
  ops and block seams ARE on main" clause was false — an overstated refusal,
  #2254's polarity — and the emitted bytes were read back out of the running hook
  to prove the repair. **THE REPAIRED CLAUSE WAS ITSELF UNBOUNDED AND IS NOW
  SCOPED**, on the fresh review's finding: it read "every op the loop composes is
  now on main", which nothing can hold, because the loop is not written and the
  set of ops it composes is therefore not yet a fact. It reads "every op #2336
  surveyed" instead — a closed population a reader can check against that issue.
  The emitted bytes were read back a SECOND time for that edit, by a temporary
  `MESSAGE` in `SUBCASE("the forward")` run with `-s`, and
  `test_qwen4_exp_scaffold.cpp` restored to its pristine sha256
  `32ee46d64b3045cbf852c387a2cb106bc46d391ed9fd19b4b6cbc013afac9b07` with `git
  diff` empty on it. The suite is unchanged at 12 / 296, which is the point
  `## Owed` already makes about it: it pins five substrings PRESENT and can pin
  none of them TRUE. **M5 is the load-bearing proof at the layer that does
  exist**, exactly as it was for W5d-1: registering the kernel on
  `DeviceType::kCUDA` instead of `kCPU` leaves the build at rc 0 and reds the
  suite BY REFUSAL, `vt: no kernel for op Qwen4ExpPleGate (id 141) on device cpu
  (type 0)`, so the dispatcher path is live rather than vestigial and the
  `op_provider.cpp` name entry is live with it.
- **The W2 host reference was NOT rerouted through the op**, the same call W5d-1
  made for `vt::RmsNormGroup` and for the same two reasons. `PleForward`'s inline
  gate loop accumulates its dot in double and is itself unreached, so routing it
  would have changed a golden-gated number and bought no reach. The two arms are
  instead held to ONE oracle: `test_qwen4_exp_ple.cpp` gates the host reference on
  section G of `qwen4_exp_ple_goldens.inc` and `test_qwen4_exp_ple_gate.cpp` gates
  the op on section J of the same file, both `exec`d verbatim out of
  transformers v5.16.0.
- **The CUDA arm of `vt::Qwen4ExpPleGate`.** Not written, for the reason W5b-3,
  W5b-4 and W5d-1 give for theirs: it could not be gated on this CPU-only host
  with no lease, and an ungated kernel is worse than an absent one. Nothing
  registers for any device but `kCPU`, so the dispatcher refuses BY NAME rather
  than falling back. It owes one decision this wave did not make for it: the CPU
  arm evaluates the divide, the clamp, the square root, the sigmoid and the
  product in DOUBLE, matching the W2 host reference term for term; a CUDA arm
  that evaluates in f32 will not inherit that bit-identity and must be gated
  against the oracle directly. It inherits ONE more obligation, added on this
  wave's review: the NaN guard in `SignedSqrt`. A CUDA arm owes its own case for
  `a NaN score PROPAGATES`, because the trap is a fall-through and not an
  arithmetic difference — every comparison in a signed square root is false for
  NaN, so any re-expression of it will swallow the NaN into `0.5 * value` unless
  it tests `isnan` first.
- **The `bf16` VALUE operand is a real value change and it is the caller's.**
  `vt::Qwen4ExpPleGate` accepts f32/f16/bf16 `value` and f32/bf16 `out`, and the
  `score` operand is f32 ONLY, for the reason `vt::SigmoidGateBf16` gives for its
  own gate: it is the argument of a sigmoid AND of a square root. The gate pins
  that a bf16 `out` is the f32 answer rounded ONCE on the store, and that a bf16
  `value` is the f32 answer recomputed on the rounded operand rather than the f32
  answer rounded. Which width the PLE block hands it is W5e-2's decision, and
  `.agents/porting.md`'s memory-format rule applies to it there.

- **THE DOT'S f32 ACCUMULATOR AT MODEL WIDTH CAN FLIP THE GATE'S SIGN, and that
  is W5e-2's decision, not this op's.** The composition this wave recommends —
  `vt::BatchedMatmul` over `[T*hc, 1, H] x [T*hc, H, 1]` views — accumulates in
  **f32** (`include/vt/ops.h`, the `BatchedMatmul` declaration: "accumulation is
  f32", and the CPU arm is the naive serial `float acc` over K).
  `test_qwen4_exp_ple_gate.cpp` runs that composition at **H = 8**;
  `Qwen4ExpTextPLELayer` runs it at **H = 2560**. The mechanism is THIS op's and
  not the matmul's: the signed square root AMPLIFIES near the origin, so a
  difference of ~1e-6 in the dot — ordinary for a 2560-term f32 reduction over
  near-cancelling operands — crosses the `|g| < 1e-6` clamp band and moves the
  gate from `+1e-3` to `-1e-3`. The SIGN is what the reduction order decides,
  and neither accumulator is more right than the other. Measured by the fresh
  review of this wave over 201 adversarial near-cancelling pairs at H = 2560,
  naive-serial-f32 against torch's pairwise f32: **68 of 201 sign flips, worst
  |gate| delta 2.0e-3, worst sigmoid delta 5.0e-4 — 50x this suite's
  `kTol = 1e-5`.** An independent reproduction in this repair flow, with a
  different construction of the adversarial pairs, read 98 of 201 flips,
  2.187e-3 and 5.467e-4 (55x), so the MAGNITUDE is the construction-independent
  part and the flip RATE is not. This is the twin of the entry W5d-1 carries for
  `vt::RmsNormGroup`'s per-group sum of squares once the group is 2560 wide
  rather than 6, and it is recorded in the same shape for the same reason.
  **W5e-2 ([#2336](https://github.com/mudler/vllm.cpp/issues/2336)) owns it**,
  because the accumulator belongs to the COMPOSITION and not to
  `vt::Qwen4ExpPleGate`, which receives a score and cannot see how it was summed.
  What W5e-2 owes is a decision — a wider accumulator for this dot, or a gate at
  model width against the oracle that pins the f32 one as sufficient — and NOT a
  tolerance widened until the flips fit. A sign flip is bimodal: it does not
  shrink with a bound, it flips or it does not.
- **W5d-1 (#2249 item 1) lands UNREACHED, by AGENTS.md "Nothing lands dead".**
  `vt::RmsNormGroup` (`include/vt/ops.h`, dispatcher `src/vt/ops.cpp`, CPU kernel
  `RmsNormGroupKernel` in `src/vt/cpu/cpu_ops.cpp`, name in
  `src/vt/op_provider.cpp`) is reached at this merge commit only by
  `tests/vt/test_ops_rms_norm_group.cpp`. No production entry point calls it:
  `ModelRegistry::Forward` is the only one this architecture has, and
  `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  before any downcast, so the PLE block that will hold the three
  `Qwen4ExpTextRMSNorm(group_size=hidden_size)` calls does not exist to call it
  from. Wiring it is owned by row `MODEL-MM-QWEN4-EXP` under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978), and gated by
  [#2249](https://github.com/mudler/vllm.cpp/issues/2249). The W2 host reference
  `qwen4_exp_ple.cpp` was NOT rerouted through the op: its `GroupedRmsNorm`
  accumulates in double by deliberate choice ("a reference choice and not a
  divergence"), it is file-local and itself unreached, so routing it would have
  changed a golden-gated number and bought no reach. **M5** is the load-bearing
  proof at the layer that does exist: registering the kernel on
  `DeviceType::kCUDA` instead of `kCPU` leaves the build at rc 0 and reds the
  suite BY REFUSAL, `vt: no kernel for op RmsNormGroup (id 140) on device cpu
  (type 0)`, so the dispatcher path is live rather than vestigial and the
  `op_provider.cpp` name entry is live with it. **M4 is a different reading and
  is recorded as one**: deleting the `RegisterOp(OpId::kRmsNormGroup, ...)` line
  outright is a BUILD refusal, `error: 'RmsNormGroupKernel' defined but not used
  [-Werror=unused-function]`, because that registration is the only reference to
  a kernel defined in the anonymous namespace `src/vt/cpu/cpu_ops.cpp` opens at
  :24. No suite runs at all under M4, so it cannot red one; what it proves is the
  link, which is why M5 exists beside it. The battery table two sections above
  states both correctly; this sentence did not, and a mutation misread as a test
  verdict is exactly the confusion the battery was run to prevent.
- **The CUDA arm of `vt::RmsNormGroup`.** Not written, for the reason W5b-3 and
  W5b-4 give for theirs: it could not be gated on this CPU-only host with no
  lease, and an ungated kernel is worse than an absent one. Nothing registers for
  any device but `kCPU`, so the dispatcher refuses BY NAME rather than falling
  back — which is the whole argument for a separate OpId over a `group_size`
  field on `RmsNormArgs`, since a new field on that shared struct would be
  silently ignored by the backends that already register `kRmsNorm` and read only
  `eps` and `gemma` off it. **SIX register it in total** — `kCPU`
  (`cpu_ops.cpp:3750`), `kCUDA` (`cuda_ops.cu:3917`), `kROCM`
  (`rocm_ops.hip:118`), `kVULKAN` (`vulkan_ops.cpp:1626`), `kMETAL`
  (`metal_ops.mm:1108`) and `kTENSTORRENT` (`tenstorrent_ops.cpp:5323`) — which
  is FIVE besides the `kCPU` this wave teaches, and five others is the number
  that carries the argument, because they are the ones that would answer a
  grouped request with a whole-row norm.

  This entry said "four", which is wrong under either reading, and the
  correction rides here because this is the paragraph the W5d-1 review already
  sent back. `include/vt/ops.h:634` says "five backends" and is NOT corrected:
  its next clause is "so a CUDA or Metal caller would get a whole-row norm
  back", so it is counting the five OTHER backends and is consistent with this
  enumeration. The two records are reconciled here rather than left to read as
  a contradiction, and no product file is touched to do it.

  The arm owes one decision this wave did not make for it: whether the per-group
  sum of squares reduces in f32 (as the CPU arm does, mirroring `x.float()` at
  `modeling_qwen4_exp.py:174` and `RmsNormKernel` beside it) or in a wider
  accumulator once the group is 2560 wide rather than 6.
- **W5d-2 (#2249 item 5): the mRoPE seam is REACHED, but only by a caller that
  is not itself routed from a production entry point.** `BuildMropeCosSinHost`
  now has external linkage behind
  `include/vllm/model_executor/models/qwen3_5_mrope.h`, and `qwen3_5.cpp`'s two
  production call sites resolve through that declaration — deleting them reds
  `test_qwen3_5_moe_vision`'s
  `qwen3_5_moe_vl_image_forward_uses_MRoPE_positions_not_plain_1d`. The hop above
  is the gap: `Qwen3_5VLGenerateGreedy`, `Qwen3_5VLGenerateGreedyVideo`,
  `Qwen3_5MoeVLGenerateGreedy` and `Qwen3_5MoeVLGenerateGreedyVideo` are DEFINED
  in `src/vllm/model_executor/models/qwen3_5.cpp:9892,9915,9960,9974` and declared
  in `qwen3_5.h` / `qwen3_5_dense.h`, and a tree-wide grep over
  `src/ include/ examples/ tools/ benchmarks/` finds no other occurrence — every
  CALLER is in `tests/`. The registered factories for
  `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`
  carry no multimodal hook, so `ModelRegistry::Forward` cannot arrive; the tree
  says so for the sibling driver at
  `include/vllm/entrypoints/openai/chat_mm.h:266-267`. This condition PREDATES
  the extraction and the extraction does not change it in either direction, but
  it is named here because `## Nothing lands dead` asks the question at every
  merge. TWO owners, because there are two ways to close it, and
  `.agents/reachability.md` asks for a row ID and an issue for each rather than
  a description:

  1. **The qwen4_exp call.** The qwen4_exp layer loop will call this seam from a
     path that IS routed through `ModelRegistry::Forward`. Row
     `MODEL-MM-QWEN4-EXP`, W5b under
     [#2031](https://github.com/mudler/vllm.cpp/issues/2031); the extraction
     itself is this row's and is tracked by
     [#2249](https://github.com/mudler/vllm.cpp/issues/2249).
  2. **Request routing to the VL drivers.** Getting an image or video request
     from the registered forward to `Qwen3_5VLGenerateGreedy` and its three
     siblings is an ENGINE seam and not this model port's. Row
     **`ENG-MM-QWEN36-VL-FORWARD`** (`.agents/engine-matrix.md`, state
     `ACTIVE`), which already owns `BuildMropeCosSinHost`, the shared
     `VLGenerateCoreGdn` and the two Qwen3.6-27B dense drivers; the two MoE
     drivers additionally sit under row
     `MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation` and
     [#891](https://github.com/mudler/vllm.cpp/issues/891). Tracked by
     [#2257](https://github.com/mudler/vllm.cpp/issues/2257), filed while
     landing this wave because nothing tracked it before: the gap is real, it
     predates the extraction, and it had no issue of its own. An earlier
     revision of this entry named this owner only as "the mm-forward row",
     which is a description and not a record.
- **W5d-4 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4) lands
  UNREACHED, by AGENTS.md "Nothing lands dead".**
  `src/vllm/model_executor/models/qwen4_exp_moe.{h,cpp}` —
  `Qwen4ExpMoeBlockWeights`, `Qwen4ExpMoeHfConfig` and `RunQwen4ExpMoeBlock` — is
  reached at this merge commit ONLY by
  `tests/vllm/models/test_qwen4_exp_moe.cpp`. No production entry point calls
  it, and the MECHANISM is not the one an earlier draft of this bullet named.
  `ForwardQwen4ExpForConditionalGeneration` EXISTS
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp:142`) and IS registered
  as the model's `.forward` hook (`:494`); its entire body is one
  `VT_CHECK(false, ...)` refusal-by-name (`:266`), placed ahead of any downcast.
  So `ModelRegistry::Forward` reaches a real hook and that hook refuses
  `Qwen4ExpForConditionalGeneration` by name before a layer runs. The conclusion
  — nothing production-side reaches this adapter — is unchanged; "the function
  does not exist" was wrong and is corrected here rather than left to be
  reasoned from. The wiring is owned
  by row `MODEL-MM-QWEN4-EXP` and tracked by
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) (the layer loop) under
  campaign issue [#1978](https://github.com/mudler/vllm.cpp/issues/1978). The
  reachability mutation is therefore vacuous here and is recorded as such rather
  than reported as a pass: deleting a production call site cannot red a gate when
  there is no production call site. What the suite DOES prove is that it enters
  through the adapter — deleting the adapter's keep-quant wiring reds it
  (mutation M4).
  Six further things W5d-4 owes:
  - **`norm_topk_prob` is not representable through this seam.** `MoeBlock`
    hardcodes `renormalize = true` (`qwen3_5.cpp:7242`) and `HfConfig` carries no
    such field, so a config that turned it off could not be honoured and the
    adapter cannot refuse what it cannot see. Upstream's default is `True`
    (`configuration_qwen4_exp.py:163`) and the released checkpoint does not
    override it, so nothing is wrong today; `Qwen4ExpParams` still does not carry
    the field, which this section already owed above.
  - **The bf16 arm is ineligible for the CUDA fast grouped-bf16 MoE.**
    `MoeBf16FastLayoutOk` (`qwen3_5.cpp:842-863`) requires per-expert `[H, I]`
    with `nk == false`, and the adapter's zero-copy views are the tower's own
    `[I, H]` with `nk == true`. It therefore falls through to the reference
    per-expert loop on CUDA, exactly as the 35B MTP producer already does. The
    alternative is a transposing copy, which is 240 GB across the stack at the
    released geometry, so this is a deliberate trade and not an oversight — but a
    grouped bf16 path that reads the tower orientation is owed if a bf16 arm ever
    becomes the shipped one. It is not one today: all seven staged checkpoints are
    quantized.
  - **The NVFP4 expert arm is refused by absence, not by name.**
    `Qwen4ExpMoeWeights` has no `Nvfp4Weight` fields and `LoadStackedExperts` has
    no fp4 branch, so `MoeBlockWeights::expert_*_fp4` are left empty and the
    seam's `fp4` predicate is false. Separately, and NOT this wave's to fix,
    filed as [#2275](https://github.com/mudler/vllm.cpp/issues/2275) and owned by
    this row:
    `LoadStackedExperts` (`qwen4_exp_weights.cpp:148-167`) handles only
    `kKeepQuant` and falls through to `ExpandBf16` for BOTH `kKeepF16` and
    `kNvfp4Fp4`, which `GgufLoadPolicy::Route` can return for
    `kStackedExpertWeight` (`gguf_keep_quant.cpp:59-60`). That silently produces a
    residency the policy did not ask for; at the released geometry it is a 240 GB
    allocation rather than a wrong answer, so it fails loudly, but it belongs to
    the W5a loader and is recorded here so the next reader does not read the
    adapter's two arms as the loader's full range.
  - **The routed top-k weights reach the experts f32, which is WIDER than the
    oracle, and the width is owed rather than defended.** Upstream casts the
    renormalized top-k weights back to the model dtype
    (`router_top_value.to(router_logits.dtype)`, `modeling_qwen4_exp.py:914`);
    our shared seam keeps them f32, because `vt::MoeRouterTopK` writes an f32
    `dtw` (`qwen3_5.cpp:7239-7242`) and every Qwen MoE in this tree reads that
    field. An earlier draft of the suite header called the seam "the more
    precise of the two". That is exactly the argument AGENTS.md §"Inherit vLLM
    defaults" exists to refuse: a token gate cannot see a dtype that is too
    wide, so "more precise" is never a reason to be wider than the oracle. The
    honest statement is that the adapter INHERITS the width from the seam, has
    no way to narrow it without diverging from every other Qwen MoE here, and
    that narrowing `dtw` to the model dtype is a seam-level change owed to the
    shared seam rather than to this adapter. Nothing measures it today: at
    top_k = 3 it is one bf16 rounding of a value in [0, 1] per pair, below this
    suite's tolerances, so it needs a gate of its own.
  - **The keep-quant arm is value-proven at Q8_0 ONLY, and against a fixture
    whose bf16 towers are `nk = false` where the loader's are `nk = true`.**
    `tests/vllm/models/test_qwen4_exp_moe.cpp` builds its keep-quant towers as
    hand-written Q8_0 blocks. The adapter is dtype-generic on that arm by
    construction — it re-declares the tower rank 2 and copies no bytes, and
    `vt::MatmulBTQuantGrouped` accepts any `IsBlockQuant` dtype — but NO k-quant
    tower is executed through it here, and every released Qwen4-Exp checkpoint
    is a k-quant (Q2_K..Q6_K, IQ1_*). A k-quant value case is owed, and it is
    owed at the loader rather than at the adapter, because it needs
    `OwnGgufQuantBlocks` output rather than a hand-built block. Separately, the
    fixture's SOURCE bf16 towers are built `nk = false` where
    `LoadStackedExperts` produces `ExpandBf16(..., /*nk=*/true)`: the adapter
    stamps `nk = true` on the per-expert views it hands the seam either way, so
    the gated bytes and the gated orientation are the production ones, but the
    fixture is not the loader's own output and this section says so rather than
    letting a reader infer that it is.
  - **The keep-quant tolerance is MEASURED, not derived.** An earlier draft of
    the suite header justified it by claiming the gate/up activation is exactly
    q8_0-representable, "amax is exactly 127*2^-8 per row". `QuantizeRowQ8_0`
    (`src/vt/cpu/cpu_quant_act.cpp:52-81`, its per-block `amax` loop at
    `:58-69`) computes `amax` per 32-ELEMENT BLOCK,
    not per row, and the fixture's `HiddenCodes()` forces `|code| = 127` at
    element 0 of each row only. `kH = 64` is TWO blocks, so block 1 takes an
    arbitrary `amax`, its `d` is not `2^-8`, and the gate and up projections
    carry quantization error as well as the down projection. The bound the suite
    asserts is therefore what it measures, not what it derives; it is printed by
    a `MESSAGE` on every run, and it is an order of magnitude below every
    mutation margin. Forcing `|code| = 127` in every block of every row would
    restore the derivation, and it is deliberately NOT done here: it moves the
    router logits, hence the routing, hence the seven mutation margins an
    independent review has already reproduced against this fixture. The
    derivation is owed to whichever change next has a reason to move the
    fixture.

- **W5b-4 (#2167) lands UNREACHED, by AGENTS.md "Nothing lands dead".**
  `vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention`
  (`include/vt/ops.h`, dispatchers `src/vt/ops.cpp`, CPU kernels
  `src/vt/cpu/cpu_qwen4_exp_qsa.cpp`) are reached at this merge commit only by
  `tests/vllm/models/test_qwen4_exp_qsa_device.cpp`. No production entry point
  calls either: `ModelRegistry::Forward` is the only one this architecture has,
  it is all-or-nothing, and `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  before any downcast. The wiring is owned by row `MODEL-MM-QWEN4-EXP` and by W5b
  under [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by
  campaign [#1978](https://github.com/mudler/vllm.cpp/issues/1978), and reaching
  the ops from the runner's caches additionally waits on
  [#2131](https://github.com/mudler/vllm.cpp/issues/2131). Mutations M24 and M25
  are the load-bearing proof at this layer: deleting either `RegisterOp` line
  reds the suite, so the dispatcher path is live rather than vestigial.
- **The CUDA arm of both QSA ops.** Not written, because it could not be gated on
  this CPU-only host with no lease, and an ungated kernel is worse than an absent
  one. Nothing registers for any device but `kCPU`, so the dispatcher refuses by
  name rather than falling back. That arm owes three decisions this wave did not
  make for it: the reduction width for the pooled key's sum of squares (`f32`
  here, in the host reference's order, because that is the order the goldens were
  dumped in); a DEVICE-side `keys_visited` counter and its copy-back, since
  `Qwen4ExpQsaAttnArgs::keys_visited` is a host pointer and cannot survive a
  launch; and whether the gather is a genuine address-generated gather on the
  device or degrades to a mask, which is the whole point of the row and is
  exactly what a CPU host cannot measure.
- ~~**W5b OWES THE INDEXER COMPOSITION IN PRODUCTION CODE, AND FOUR SETTINGS WITH
  IT.**~~ **DISCHARGED by W5b-5 ([#2211](https://github.com/mudler/vllm.cpp/issues/2211)),
  and the VALUE gate it demanded exists.** `Qwen4ExpQsaIndex`
  (`src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp`) composes the three
  ops and states all four settings at one site, each beside the upstream line it
  mirrors and the mutation that reds it. The gate is
  `tests/vllm/models/test_qwen4_exp_qsa_block.cpp`, against the oracle's OWN
  pre-top-k `scores` tensor: fed `k...IdxQPost` and `k...IdxKRaw` — the oracle's
  own roped query and raw keys — the composed logits are BIT-IDENTICAL to it,
  max abs **0** over scales of 3.37 and 6.24 across 12 and 60 logits. M1
  (inherit `n_head ** -0.5`) and M2 (drop the softmax scale) both RED on that
  case, which is the repair for M26's recorded survival. The paragraph below is
  kept because it is the reasoning the repair rests on, not because the debt is
  open. The composition WAS in exactly one place, the `RunIndexer` helper in
  `tests/vllm/models/test_qwen4_exp_qsa_device.cpp`, and nothing under `src/`
  enforced any of the four settings the collapse depends on:
  1. `weights` is all ones (`[T, index_n_heads]`), which is what collapses the
     per-head fold to a single constant. M27 is its red control.
  2. `n_head_scale == 1.0f`, NOT DeepSeek-V4's `n_head ** -0.5`, which QSA has no
     tensor for.
  3. `softmax_scale == index_head_dim ** -0.5`, QSA's own scale.
  4. `win_end == kv_len / compress_ratio` per query token — the COMPLETE visible
     blocks, not the whole cache. M28 is its red control.
  Two of the four have no gate that would catch a wrong value in production:
  M26 records that `n_head_scale` is invisible to selection BY CONSTRUCTION,
  because top-k is invariant under a positive rescale of every score, and
  `softmax_scale` is invariant for the same reason. That is why the W5b-5 gate is
  a VALUE gate on the logits and not a selection gate.
- **A single-pass online softmax for the gather.** The CPU kernel makes two
  passes over the selected rows per query head, which is why the honest read
  count is `selected * num_q_heads * 2`. A single-pass rewrite legitimately
  halves it, and `kReadsPerRowPerHead` in the device suite is where that constant
  gets re-derived on purpose rather than silently absorbed. No speed claim is
  admissible from this row until G2 passes, so this is owed, not deferred work.
- **The `bf16` operand arms of both ops are declared and UNGATED.** The
  dispatchers accept `f32`/`bf16` and the kernels widen through `LoadF32At`, but
  every fixture is `f32`-valued and the goldens are `f32` arrays of
  bf16-representable numbers, so no case stores a bf16 tensor. That is honest
  rather than complete: the `round_intermediates_to_bf16` flag is gated (M7, M9),
  the bf16 STORAGE path is not.
- **`Qwen4ExpQsaCompress` assumes a CONTIGUOUS visible prefix**, as the W4 host
  reference does. Upstream forms blocks over `local_visible_indices` of a padded
  batch; a serving engine's ragged batch has no interior masking, so the two
  coincide and block `b` is exactly tokens `[CR*b, CR*b + CR)`. The op REFUSES a
  key count that is not a whole number of complete blocks (M20 reds that refusal)
  but it cannot detect an arbitrary visibility set, and nothing yet does.
- **The side cache's paged store — READ THE SCOPE, W5i DID NOT CLOSE THIS ONE.**
  This entry is about `vt::Qwen4ExpQsaCompress`'s OUTPUT, the POOLED block keys,
  and not about the raw indexer keys W5i paged. Upstream caches no pooled key at
  all — it rebuilds them from the raw keys every step
  (`modeling_qwen4_exp.py:679-682`) — so this array is scratch by construction
  and a paged home for it is an optimisation, not a correctness gap.
  `QsaSideCacheSpec` (W4) says what the cache
  costs and `QsaCompressedSlot` says which slot a token writes; this op writes a
  DENSE `[num_blocks, head_dim]` array and not a paged one. The block-table store
  belongs to the wave that gives QSA a real KV-cache group, which is blocked on
  [#2131](https://github.com/mudler/vllm.cpp/issues/2131). **This is a PRODUCTION
  obligation only.** The wave's first draft also recorded the fetch-level PROOF —
  a cache whose unselected blocks fault when touched — as waiting on the same
  store. It never was: the fresh review built it out of `mmap` and
  `mprotect(PROT_NONE)` inside the test, it is the case `the gather never FETCHES
  an unmapped unselected row`, and M11c is the paired control showing it convicts
  a body the NaN poison cannot see. Nothing about the instrument is owed.

- **W5b-5 ([#2211](https://github.com/mudler/vllm.cpp/issues/2211)) lands
  UNREACHED, by AGENTS.md "Nothing lands dead".**
  `src/vllm/model_executor/models/qwen4_exp_qsa_block.{h,cpp}` — `Qwen4ExpQsaIndex`
  and `RunQwen4ExpQsaBlock` — are reached at this merge commit only by
  `tests/vllm/models/test_qwen4_exp_qsa_block.cpp`. No production entry point
  calls either: this architecture's only one is `ModelRegistry::Forward`, it is
  all-or-nothing, and `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  because the LAYER LOOP is not written. The wiring is owned by row
  `MODEL-MM-QWEN4-EXP` under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978). Also owed from that
  wave, each named rather than discovered later:
  - **THE LAYER LOOP, WHICH IS NOW THE ONLY THING BETWEEN THIS ROW AND A TOKEN.**
    Every seam and every op the forward needs is on `main` — `RunGdnBlockPaged`
    (W5b-1), the two gated-residual ops (W5b-2), `vt::Qwen4ExpPleConv` (W5b-3),
    the two QSA ops (W5b-4) and now the QSA BLOCK (W5b-5). What has no production
    shape yet is: the PLE block (the n-gram hash and its gather composed with
    `vt::Qwen4ExpPleConv`), the GDN and MoE weight adapters
    (`Qwen4ExpGdnWeights` -> `GdnLayerWeights`, `Qwen4ExpMoeWeights` ->
    `MoeBlockWeights`; the GGUF loader mirrors `qwen3_5_gguf_weights.cpp` name for
    name and shape for shape, so the adapters are field aliasing rather than
    arithmetic), the 10240-wide hyper-connection stream through the per-layer
    loop, the interleaved-mRoPE cos/sin table build, and the terminal mixer plus
    `lm_head`.
  - **THE `hc_norm` POLARITY IS A TRAP THE LAYER LOOP WALKS INTO, and it is
    recorded here because W5b-5 hit the same shape and got it right by accident
    of scope.** `LoadNormBf16(..., unshift=true)` INVERTS the converter's baked
    `+1`, so every gamma the loader stores is the RAW HuggingFace value, centred
    on 0. `vt::Qwen4ExpQsaCompress` wants exactly that and applies `(1 + w)`
    itself; `vt::RmsNorm` wants it under `RmsNormArgs::gemma = true`, which is
    what this block passes and what mutations M9/M10/M11 red. But
    `vt::Qwen4ExpGatedResidual` documents the OPPOSITE convention — "hc_norm_w is
    vLLM's parameterization, i.e. ALREADY `1 + w_hf` … This op never adds 1" —
    so the layer loop must fold `hc_norm`, `norm_key`, `norm_query` and
    `norm_conv` with `vllm::qwen4_exp::HcNormWeightFromHf` before handing them to
    that op. Miss it and every gated residual applies a near-zero scale, which
    reads as a checkpoint bug rather than a port bug. Nothing gates this today,
    because the layer loop is the first caller. **Tracked as
    [#2218](https://github.com/mudler/vllm.cpp/issues/2218)**, which is its own
    wave and deliberately not repaired here. The contradiction is visible AT THE
    LOAD SITE and does not need the op to be read to be seen:
    `qwen4_exp_weights.cpp:258-263` argues FOR the fold in its own comment — "the
    fold is what makes the file's value the multiplier our own `out * weight`
    grouped norm wants", corroborated elementwise on three published artifacts —
    immediately above the line that strips it with `unshift=true`.
  - ~~**The PAGED cache.** This block takes CONTIGUOUS per-sequence K/V and a
    [#2131](https://github.com/mudler/vllm.cpp/issues/2131) and on W5c.~~
    **HALF DISCHARGED by W5d-3 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249)
    item 2)**, and the half is named so nobody reads this as done. The QSA
    layers' K/V — KV group 0, the `FullAttentionSpec` — now has a paged consumer:
    `Qwen4ExpQsaPagedCaches` + `RunQwen4ExpQsaBlockPaged`, over a
    `kv_block_table`/`kv_block_size` ADDRESS MODE inside the same
    `vt::Qwen4ExpQsaGatherAttention` rather than a second op. The INDEXER side
    cache was still contiguous at W5d-3 — that is KV group 2, the
    `MLAAttentionSpec`, and its paged store was the separate entry above and
    #2249 item 3 — and **W5i closed it**, by composition rather than by a second
    address mode: `vt::IndexCopy` for the store, `vt::IndexSelect` for the read,
    no op touched. What
    W5d-3 did NOT need from #2131 is worth recording, because this bullet asserted
    the dependency for three waves: the K/V paged read needs only a block table and
    a slot mapping, both of which the runner already builds for every full-attention
    model, and none of the multi-state recurrent work #2131 owns.
  - **The RAGGED-BATCH form.** `kv_lens[t] = past_len + t + 1` is built inside the
    block from a CONTIGUOUS visible prefix. Upstream's general form reads an
    arbitrary visibility set out of a padded batch's mask, and the ops' own
    `## Owed` already records that nothing here can detect one; the block
    inherits that limit rather than adding to it.
  - **The cos/sin table BUILD**, and with it the interleaved-mRoPE section
    layout. The block takes the tables as operands in BOTH layouts the two ops
    want — a bf16 PACKED `[P, rot]` cos|sin cache for `vt::RopeFromCache` and two
    f32 FULL `[P, rot]` tables for `vt::Qwen4ExpQsaCompress` — and asserts each by
    name. It also CROSS-CHECKS the two against each other, which it did not when
    the header first claimed it did: equal heights, then a BOUNDED SAMPLE of rows
    (row 1, the midpoint and the last row) compared value for value at one bf16
    ulp. Row 0 is not a probe, because cos is 1 and sin is 0 at every frequency
    there and it agrees under every construction difference. What the sample
    cannot see is a single corrupted row; what it does see is every table-wide
    difference a layer loop can make — a different theta, a different
    `rotary_dim`, an interleaved pack, swapped halves, an off-by-one position
    offset, or a position scaling applied to one table and not the other. A FULL
    comparison is deliberately not paid: it would be O(P * rot) per QSA layer per
    step to re-check a constant. The wave that builds them still owes the case
    with three DISTINCT position streams this spec already records as unowned.
  - **The bf16 STORAGE arm is the only arm.** The block refuses an f32 `hidden`
    by name, and the reason is a shared-surface fact rather than a preference:
    every `vt::` output-gate op in this tree — `SigmoidGateBf16`,
    `SharedExpertGate` — stores bf16 on every backend, because vLLM resolves one
    model dtype and this tree inherits that polarity. An f32 arm would have to
    widen a dispatcher across five backends this host cannot gate, and the
    refusal says so.
  - **The CUDA arm**, which is the QSA ops' own owed item and not a new one. The
    block adds no arithmetic, so it inherits that debt unchanged. TWO things ARE
    new at the BLOCK level, and they are named here rather than folded into that
    inherited debt, because a device arm has to answer both and neither is
    visible from the ops:
    - **The indexer's per-call INDEX BUILD is done on the host.**
      `Qwen4ExpQsaIndex` materialises `ones` `[T, H]`, `win_start` `[T]` and
      `win_end` `[T]` into host vectors and hands each to a `DBuf`, and
      `RunQwen4ExpQsaBlock` does the same for `kv_lens` `[T]` — FOUR small
      host-to-device copies per QSA layer per step on a device queue. It also
      `VT_CHECK`s that `kv_lens` is CPU-resident and reads it on the host to build
      the window, which is a refusal a device-resident batch would hit by name.
      Upstream rebuilds exactly the same metadata on every call —
      `local_visible_indices` out of `torch.nonzero` on the mask row,
      `block_token_indices`, `group_starts` and `selected_token_indices`, inside a
      `for batch_idx / for query_idx` Python loop
      (`modeling_qwen4_exp.py:667-702`) — so the PER-CALL REBUILD is a faithful
      mirror rather than a divergence. What is not inherited is the transfer:
      upstream has one device and no H2D edge to pay, so the device arm owes the
      decision of where these four are built, and the nested Python loop is a
      reminder that the oracle is a reference implementation and not a
      performance model.
    - **The pooled BLOCK KEYS are recomputed over the ENTIRE cache every step.**
      The `block_keys` scratch is allocated per call and dropped, and
      `vt::Qwen4ExpQsaCompress` runs over cache rows `[0, complete_keys)` — O(kv)
      per layer per token, for a quantity that only ever GROWS by one block every
      `compress_ratio` tokens. Upstream does the same, and worse: it recomputes
      `pooled_keys` and `block_key_states` inside the per-query-token loop
      (`:679-686`), so its cost is O(kv) per query token per layer. So this is a
      faithful mirror of a reference implementation, and it is the shape that
      makes the incremental store worth having. The wave that gives QSA a real
      KV-cache group turns this scratch into the side cache's paged store and
      inherits the choice of whether to keep it incremental.
  - **The FETCH-level proof is inherited, not re-built.** The `mprotect(PROT_NONE)`
    unmapped-tail probe lives one layer down in
    `test_qwen4_exp_qsa_device.cpp`, and it is load-bearing for this block because
    the block's ONLY consumer call is `vt::Qwen4ExpQsaGatherAttention`. Mutation
    M20 is what says that call is the one under test: handing the consumer every
    VISIBLE block — a dense walk wearing a gather's clothes — reds 3 of 8 cases
    and 130 assertions.

- **W5d-3 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 2) lands
  UNREACHED, by AGENTS.md "Nothing lands dead".** The paged QSA consumer —
  `Qwen4ExpQsaPagedCaches` and `RunQwen4ExpQsaBlockPaged`
  (`src/vllm/model_executor/models/qwen4_exp_qsa_block.{h,cpp}`) together with the
  `kv_block_table`/`kv_block_size` address mode on
  `vt::Qwen4ExpQsaGatherAttention` — is reached at this merge commit only by
  `tests/vllm/models/test_qwen4_exp_qsa_block.cpp`. The reason is unchanged from
  W5b-5 and is not a property of this wave: this architecture's only production
  entry point is `ModelRegistry::Forward`, it is all-or-nothing, and
  `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  because the LAYER LOOP is not written. The wiring is owned by row
  `MODEL-MM-QWEN4-EXP` under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978). The reachability
  mutation `.agents/reachability.md` prescribes has no site here for the same
  reason it had none for W5b-5: there is no production call site to delete.
  Also owed from this wave:
  - **The INDEXER side cache was still contiguous — CLOSED by W5i.** #2249 item
    3 — KV group 2 is never gathered — was owed as W5c-2 and deliberately not
    smuggled into this wave. At W5d-3 `Qwen4ExpQsaPagedCaches::index_key` was
    `[max_kv, indexer_head_dim]`, so a forward built on this arm still needed a
    contiguous side cache per sequence. It is now the fused MLA page
    `[num_pages, block_size, indexer_head_dim]` with group 2's own block table.
  - **ONE REQUEST PER CALL.** `kv_block_table` is `[1, max_pages]` and the block
    refuses anything else by name. A ragged multi-request batch needs the
    `query_start_loc` plumbing `vt::PagedAttention` carries and this block does
    not, on top of the RAGGED-BATCH `kv_lens` item already owed above.
  - **An fp8 paged KV cache is REFUSED BY NAME.** `vt::Qwen4ExpQsaGatherAttention`
    has no dequantising read and no `k_scale`/`v_scale`, so an fp8 page would be
    read as floats — wrong tokens, not a crash, which is the exact failure
    `kv_cache_route.h` exists to prevent. The refusal is gated.
  - **The CUDA arm of the paged address mode**, inherited from the QSA ops' own
    owed CUDA arm and not a new debt: the address resolution is four lines of
    integer arithmetic in the same kernel body, so whatever answers the ops
    answers this.

- [#1978](https://github.com/mudler/vllm.cpp/issues/1978): this port, the campaign
  row. W0 landed the spec with no product code.
- [#1981](https://github.com/mudler/vllm.cpp/issues/1981): **W1**, the config
  surface — resolution, validation, registration, refuse-by-name on everything
  else. LANDED. Recorded here because every `Refuse()` message this code emits
  ends "See `.agents/specs/qwen4-exp-flash-next.md` and issue #1981", and a reader
  who follows that pointer has to find the issue at the other end of it.
- **`Qwen4ExpParams` resolves 60% of the config.** `linear_num_key_heads`,
  `linear_num_value_heads` (48 in the checkpoint, against upstream's declared
  default of 32 — a difference W2 must not inherit from the docstring),
  `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim`,
  `norm_topk_prob`, `max_position_embeddings` and the resolved `output_gate_type`
  are read by the shared `HfConfig` and dropped by this struct. Nothing is lost
  yet; W2/W3 owe carrying the ones they consume.
- **A model-layer oracle.** The config layer is gateable and now gated
  (`## The refusal boundary`); nothing above it is. `gateable = no` stands.
- GGUF k-quant arms, including authoring the `qwen4_exp` architecture on our side,
  and the statement that no llama.cpp oracle exists for them.
- MTP depth > 1.
- **W2 (#1987) lands UNREACHED, by AGENTS.md "Nothing lands dead".**
  `src/vllm/model_executor/models/qwen4_exp_ple.{h,cpp}` is a host reference
  for the n-gram hashed embedding and the PLE dilated depthwise conv. No
  production entry point calls it: `qwen4_exp` has no registry entry, no
  loader and no `ModelRegistry::Forward` arm until W5 assembles the model.
  The wiring is owned by row `MODEL-MM-QWEN4-EXP` (W5) and tracked by
  campaign issue [#1978](https://github.com/mudler/vllm.cpp/issues/1978).
  Also owed from that wave: the batched device arm (the host signatures are
  per-sequence precisely so it drops in), the 128-shard NUMERIC table
  reassembly, and the prefix-caching decision for a conv state written by a
  chunked prefill shorter than 9 columns, which `## Design` records as
  AMBIGUOUS and not resolvable from upstream.
- **W5b-3 ([#2156](https://github.com/mudler/vllm.cpp/issues/2156)) lands
  UNREACHED, by AGENTS.md "Nothing lands dead".** `vt::Qwen4ExpPleConv` and
  `src/vt/cpu/cpu_qwen4_exp_ple.cpp` are reached only from
  `tests/vllm/models/test_qwen4_exp_ple_device.cpp`. No production entry point
  calls them: the architecture's only one is `ModelRegistry::Forward`, which is
  all-or-nothing, and it has no `qwen4_exp` arm. The wiring is owned by row
  `MODEL-MM-QWEN4-EXP`, tracked by
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) (the forward) under
  campaign issue [#1978](https://github.com/mudler/vllm.cpp/issues/1978).
  Also owed from that wave:
  - **The CUDA arm.** Not written, because it cannot be gated on a CPU-only
    host, and an ungated kernel is worse than an absent one. It inherits one
    decision: this CPU kernel accumulates its four taps in `double` and the
    device gate asserts BIT-IDENTITY with the W2 host reference at the model's
    10240-channel width. An f32-accumulating CUDA kernel does not inherit that
    identity — mutation M9 measures exactly this — so it must either accumulate
    wider or be gated against the pinned oracle directly.
  - **A bf16 `conv_state`.** The dispatcher refuses one by name.
    `CausalConv1dSpecUpdate` admits bf16 state on CUDA because a CUDA kernel
    there writes it; here nothing does, and admitting a dtype no arm can produce
    would be a promise with no kernel behind it. It is owed with the CUDA arm.
  - **Reaching the op from the runner's recurrent cache.** The op takes its state
    as an explicit `[N, C, (K-1)*dilation]` operand plus a per-sequence row
    index. That parameter is called `conv_state_indices`, after
    `CausalConv1dUpdate`'s parameter for the same axis, and it is deliberately
    NOT spelled `state_idx`, because upstream's `state_idx` selects
    one of a PLE layer's three states, and those three cannot be planes of one
    tensor because `cache_utils.py` keeps `conv_states` as a list with a
    per-entry `conv_kernel_size[state_idx]` and the widths are 4, 9 and 2, over
    different channel counts and, for the third, over integers. Resolving that
    selector is the caller's job, and the caller cannot exist until
    [#2131](https://github.com/mudler/vllm.cpp/issues/2131) generalises the
    runner's one-group/two-shape `MambaSpec`.
  - **The prefix-caching decision** for a conv state written by a chunked prefill
    shorter than nine columns, which `## Design` records as AMBIGUOUS and not
    resolvable from upstream, is untouched by this wave. This op reproduces
    upstream's zero-pad exactly; it does not decide what a cache HIT should
    restore.
- **W2's float path has never been compared at MODEL WIDTH, and that is the one
  gap its own gate cannot close.** `tests/vllm/models/test_qwen4_exp_ple.cpp`
  runs at `hidden_size = 8`, `hc_count = 2`, `heads_per_ngram = 2`,
  `ngram_vocab_size_base = 20`. Only the multipliers, the prime head sizes and
  the offsets are pinned at the released config, and those are INTEGERS, where
  width cannot change an answer. Everything float — the grouped RMSNorm, the
  gate reduction that is 2560 wide in the real model, the 10240-channel dilated
  conv — is gated at width 16 with 8-wide groups. Every structural mutation in
  the W2 table dies there by orders of magnitude, so the instrument is sound for
  structure; a REDUCTION-ORDER difference at width 2560 is what it cannot see,
  and it is exactly the class of difference that a device arm introduces.
  Owed: a first real-width numeric comparison against the lane pin. It must
  derive a **relative** bound, not reuse W2's absolute `1e-5`. W3's repair on
  the sibling branch measured the reason: an exact-double evaluation of the
  oracle's own algorithm for the gated residual already exceeds a 1e-5 absolute
  bound at model width, because torch runs the reduction in fp32, so an absolute
  bound at that width tests the accumulator and not the port.
- **CLOSED by W5e-2 ([#2336](https://github.com/mudler/vllm.cpp/issues/2336)):
  the `conv_mask` PAIRED obligation now has an enforcer.** The entry this
  replaces said W2 gates the masking itself (both tensors, and through the
  9-column state) but that the paired half — a masked position must already
  carry EOS in `input_ids`, because the hash reads ids and not activations — was
  a CALLER obligation with no caller. `RunQwen4ExpPleBlock` is that caller and it
  refuses the pair BY NAME rather than documenting it, which is the first
  enforcer this half has had anywhere in the tree. Spec mutation M7 deletes the
  refusal and reds. What is NOT closed is one hop further out: the block has no
  production caller, so nothing yet BUILDS a `conv_mask`, and the wave that does
  (W5f, the layer loop) must hand this block a mask and ids that already agree —
  which is now a refusal it will hit rather than a paragraph it may miss.
- The 1M-token RoPE extension above the native 262144.
- The non-resident n-gram table on CUDA. **W6a (#1989) discharged the CPU half**:
  the dequantizing gather (`vt::Embedding` over a block table) and the
  `kEmbeddingTable` keep-quant policy change both landed, gated bit-exactly
  against llama.cpp `b10451` decoding real bytes of the shipped tensor. What is
  still owed is the **CUDA arm**: `EmbeddingKernelCuda` (`src/vt/cuda/cuda_ops.cu`)
  refuses anything but f32/bf16, so `DeviceQuantGatherSupported` returns false on
  CUDA and the table keeps its expand-bf16 residency there. That is the honest
  state and it is also the expensive one — a device-resident quantized table
  gathered on device is precisely the shape llama.cpp's #27742 does NOT have (it
  pins the table to the CPU by tensor class), so the CUDA arm is where this
  model's high-concurrency advantage lives, not a tidying task. Still owed with
  it: a measurement of the page-cache cost that the <= 64 KiB/token arithmetic
  only bounds.
- **VERIFIED 2026-08-26, no longer owed:** llama.cpp's substitution for a
  ragged-K tensor is read at the pin, `src/llama-quant.cpp:374-405 @ b10451`
  (`tensor_type_fallback`). `Q4_K -> Q5_0` is confirmed exactly as this spec
  asserted, and `IQ4_XS -> IQ4_NL` beside it, which is why the shipped UD-IQ1_S
  carries 49 IQ4_NL tensors. Both encodings landed in W6a.
- **NEW, from reading that table:** the same function maps `Q5_K -> Q5_1` (ggml
  type 7) and `Q2_K`/`Q3_K` -> `Q4_0`. Q5_1 and Q4_1 (3) are still absent from
  our reader, so a `-Q5_K_M` recipe of THIS model — whose `ffn_down_shexp` row is
  640 and therefore ragged for any K-quant — would refuse at header parse. Not
  in W6a's scope, which the shipped file does not need; recorded rather than
  quietly added.
- **A keep-quant gather for `deepseek4` and `laguna`.** W6a made
  `GgufTensorRole::kEmbeddingTable` keep-quant eligible, which is a change to a
  SHARED policy with three consumers. Only `qwen3_5_gguf_weights.cpp` was given
  the residency; `deepseek_v4_weights.cpp` and `laguna_weights.cpp` consume
  `token_embd` as a flat host f32 array (and, on a tied file, hand the same f32
  image to the final projection), so both now narrow the policy for that tensor
  through `NoKeepQuant` and keep expanding it. That is correct and it is not
  free: on a real deepseek4 or laguna checkpoint the vocab matrix is still
  materialized in f32. Decoding it per gathered row instead needs those two
  forwards to take a `vt::Tensor` rather than a `std::vector<float>`, which is
  model work and not policy work. Owed to
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978).
- The non-resident n-gram table on CUDA: the dequantizing gather op and the
  `kEmbeddingTable` keep-quant policy change (Route B), and a measurement of the
  page-cache cost that the <= 64 KiB/token arithmetic only bounds.
- ~~llama.cpp's ragged-K substitution~~ **RESOLVED, AND NOW READ AT THE PIN**:
  `Q4_K -> Q5_0`, `IQ4_XS -> IQ4_NL`, from `tensor_type_fallback` in
  `src/llama-quant.cpp:374-406` of the `llama-cpp` oracle at its recorded revision
  `10bf611e533d81f739128304991c5e133c6aebd8` (`b10451`,
  [`../oracles/llama-cpp.md`](../oracles/llama-cpp.md)) — not at `master`, which is
  where the claim was first read and which is not an oracle. The complete table at
  that revision: `IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ4_XS
  -> IQ4_NL`; `Q2_0`/`Q2_K`/`Q3_K`/`TQ1_0`/`TQ2_0 -> Q4_0`; `Q4_K -> Q5_0`;
  `Q5_K -> Q5_1`; `Q6_K -> Q8_0`; anything else throws. Both are reachable for this
  model depending on the recipe, and our reader supports NEITHER (no `case 6`, no
  `case 20`), so W6 owes both.
- **A published GGUF now EXISTS**, which supersedes this spec's "no GGUF exists and no
  tool can produce one": `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, 67.56 GiB of
  weights in 3 shards, `general.architecture = qwen4exp`, 1224 tensors. **PINNED**, and
  it needed to be — the repo's `lastModified` moved to `2026-08-26T15:54:43Z`, after
  W1's pull request was opened, which is exactly the re-quantize-in-place case AGENTS.md
  "Say which weights, and from where" names. Revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`; at that revision, shard sizes
  10,946,624 + 49,990,818,368 + 22,544,696,352 = **72,546,461,344 bytes = 67.564 GiB**,
  with sha256 `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`,
  `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6` and
  `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a`. Those digests were
  the Hub API's `lfs.oid` values; a LOCAL sha256 was owed and is now recorded. All three
  were recomputed on the staged copy on 29 August 2026 and agree three for three, in
  [the ladder-arm evidence file](../../docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md). The "1224 tensors" count remains UNVERIFIED: shard 1 is the
  metadata shard and reports `n_tensors = 0`. It FITS GB10
  with ~52 GiB of headroom, and two things in OUR tree stop us loading it: the missing
  IQ4_NL reader arm, and the gather-table expansion. Its metadata independently
  confirms this spec's n-gram derivation to the digit --
  `ple.layer_multipliers = [23703573157769, 20109073645365, 8052911324071]` and
  `ple.head_vocab_sizes = [20000003, 20000023, ...]`.
- **Mirror the `qwen4exp` GGUF key and tensor names rather than inventing ours.** Two
  competing llama.cpp PRs (#27742 open, #27739 closed-by-courtesy) already disagree on
  `ple.*` key spellings and on whether the n-gram table is model-level
  (`per_layer_token_embd`) or per-layer (`blk.N.ple_ngram_embd`), and a maintainer has
  asked for a rename, so the names are NOT settled. Re-check before W6a commits to a
  layout; a wrong guess makes every published GGUF unreadable by us.
- A K-divisibility assertion in whatever writes our GGUF files.
- A speed denominator, once one exists.
- **W4's QSA slice lands UNREACHED**, and this entry is what AGENTS.md "Nothing
  lands dead" requires in exchange.
  `src/vllm/model_executor/models/qwen4_exp_qsa.{h,cpp}`
  ([#1991](https://github.com/mudler/vllm.cpp/issues/1991)) ship the indexer, the
  side-cache sizing and the GATHER consumer as host reference math with no
  production call site: `Qwen4ExpTextModel` does not exist yet, its PLE
  ([#1987](https://github.com/mudler/vllm.cpp/issues/1987)), hyper-connection
  stream ([#1988](https://github.com/mudler/vllm.cpp/issues/1988)) and GGUF
  reader ([#1989](https://github.com/mudler/vllm.cpp/issues/1989)) are sibling
  waves, and the registry entry plus runner wiring belong to W5. Row
  `MODEL-MM-QWEN4-EXP` owns that wiring and
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978) tracks it.
- **The QSA device arm.** `qwen4_exp_qsa.cpp` is the portable oracle a CUDA
  kernel is written against, the way `deepseek_v4_dsa.h` is for
  `src/vt/cuda/cuda_deepseek_v4.cu`. Nothing in W4 runs on a GPU, so the gather's
  cost advantage over the mask is stated by a `keys_visited` count and NOT by a
  measurement. The speed axis opens at G4.
- **`QsaCompressNormRope` assumes a contiguous visible range.** Upstream forms
  blocks over `local_visible_indices` of a padded batch; a serving engine's
  ragged batch has no interior masking, so the two coincide and the function
  asserts `num_keys % compress_ratio == 0` instead of accepting an arbitrary
  visibility set. A padded-batch caller would need the general form.
- **The row's lifecycle record is owed the W4 transition, and W5 lands it.** W4
  ([#1991](https://github.com/mudler/vllm.cpp/issues/1991)) is this row's first
  product code: `src/vllm/model_executor/models/qwen4_exp_qsa.cpp` joins
  `add_library(vllm ...)` at its merge commit. `.agents/model-matrix.md` still
  carries the row at `READY` with the note "SPEC ONLY, NO PRODUCT CODE, NO TOKEN,
  NO SPEED", which was true at the merge base and is false from W4 onwards. That
  cell is NOT edited here: W1 through W3 are live on the same file and the
  operator is sequencing those writes, and a per-wave edit to one shared row is
  exactly the lock AGENTS.md "Records" forbids. W5, which lands the registry entry
  and the runner wiring, moves the row to `ACTIVE`, rewrites that note and updates
  `## Now` in the one change. Until then this entry is where the discrepancy is
  visible.
- **Nothing gates the interleaved-mRoPE section layout, in W4 or anywhere yet.**
  `gen_qwen4_exp_qsa_goldens.py` passes a 2-D `position_ids`, which
  `Qwen4ExpTextRotaryEmbedding.forward` expands into three IDENTICAL streams, so
  `apply_interleaved_mrope` runs value-blind and the captured `cos`/`sin` are
  indistinguishable from plain RoPE. `qwen4_exp_qsa.h` scopes the tables out of W4
  ("this function does not build them") and W4 is honest about that, but no wave
  currently owns building them, and a multimodal caller with genuinely different
  t/h/w streams would be running an untested section layout. The wave that builds
  the cos/sin tables owes a case with three DISTINCT position streams.
- **W3's host reference lands UNREACHED, and this is the record of it** per
  AGENTS.md "Nothing lands dead".
  `src/vllm/model_executor/models/qwen4_exp_hc.{h,cpp}`
  ([#1988](https://github.com/mudler/vllm.cpp/issues/1988)) is reached only by
  `tests/vllm/models/test_qwen4_exp_hc.cpp`. No production entry point calls it
  at its merge commit: W1 config registration
  ([#1986](https://github.com/mudler/vllm.cpp/issues/1986)) was still in review,
  so no `qwen4_exp` resolves through the loader and there is nothing for the
  gated-residual stream to hang off. The wiring is owed by **W5, assembly**,
  under [#1978](https://github.com/mudler/vllm.cpp/issues/1978), which is the
  wave that widens the residual buffers to `hc_count * hidden_size` and calls
  the module twice per layer.
- The **model-matrix lifecycle cell** for
  `MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation`, which still reads
  `SPEC ONLY`. Left to W1 deliberately rather than by omission: W1 is the wave
  whose scope IS registration, its pull request is already open, and
  `.agents/model-matrix.md` is a single shared file, so three parallel waves
  editing one cell is the write-lock AGENTS.md "Records" names. Whichever of
  W1/W2/W3 lands last owes the correction.
- The **device arm of the gated residual**, and with it one check this host wave
  cannot make: that `RMSNormGated.forward_cuda`'s flash-linear-attention Triton
  kernel is numerically correct in its GROUPED mode (unverified upstream, see
  `## Design`).
- **W3's `kTol = 1e-5` is an absolute bound that does not survive a rescale, and
  the host reference is the first thing it fails.** Recorded because an earlier
  draft of the bullet above framed the tolerance question as the DEVICE arm's
  problem, and it is not. Measured against the pinned oracle itself, at the
  model's own shape (hidden_size 2560, hc_count 4, hc_lowrank 320, eps 1e-6, two
  tokens), max|diff| on `mixed_input`. This is ONE draw of random inputs, and the
  ratios below move from draw to draw; the ordering and the conclusion do not.

  | | t=0 | t=1 |
  |---|---|---|
  | ours (fp32) vs oracle | 2.325e-05 | 2.137e-05 |
  | exact double vs oracle | 1.360e-05 | 5.431e-06 |
  | ours (fp32) vs exact double | 3.684e-05 | 1.606e-05 |

  At the suite's own widths (flat = 24 and 15) the implementation is bit-identical
  to the oracle -- max|diff| over every golden array of cases A, B and C is
  2.384e-07 -- so kTol carries a 42x margin there and constrains nothing. At model
  width our fp32 interior is 2.1x to 2.3x over it, driven by `LinearNoBias`'s
  sequential fp32 accumulation over 10240 terms. **The second row is the one that
  settles it: the ORACLE is itself of the same ORDER as kTol against an exact
  evaluation of its own algorithm -- 1.36x on the draw above, 0.91x and 0.82x on
  an independent draw taken during fresh review -- because torch runs this in
  fp32 too.** No fp32
  implementation of this function meets a 1e-5 ABSOLUTE bound at hidden_size
  2560, and widening our accumulator cannot rescue one. W5 therefore does not
  reuse kTol at model width; the file carries a real-width case with a relative
  bound (`kRealWidthMixedRel`, 4e-5, derived as 6.6x the sqrt(K)*u random-walk
  bound for K = 10240) that all three measurements sit inside. **What is still
  owed** is agreement with the ORACLE at model width, which needs a real
  checkpoint and cannot be closed in-suite: the in-suite case compares against
  the double reference, because dumping one token of oracle IO at this width is
  26 MB of `.inc`.
- **The double accumulator is now gated, and the device arm inherits the
  consequence.** `GroupedRmsNorm` accumulates the per-group sum of squares in
  `double`, and at the suite's group sizes of 5 and 6 that convention had zero
  discriminating power -- replacing it with `float` left the suite 280/280 green.
  It is gated at the model's real group size of 2560, on magnitude-separated
  data, where the two accumulators differ by 742x (3.168e-06 against 2.352e-03,
  bound 1e-4). The convention is kept rather than dropped because it makes the
  host reference more accurate than the oracle rather than less, which is what a
  reference is for. What follows for the device arm, stated here so it is not
  discovered: **a straight fp32-accumulate device reduction will not meet
  `kRealWidthNormTol` on that data.** That is the correct signal, not a defect in
  the gate -- it says the device kernel must accumulate wider than fp32 or be
  gated against the oracle directly rather than against this reference. Deciding
  which is the device wave's, and it is owed.
- The **fused rank-1 write-back**. `GatedResidualWriteBackInPlace` is the seam
  and is already the primitive, but no device kernel replaces it yet. Both
  llama.cpp implementations of this architecture materialise the update as a
  `repeat_4d` + `mul`, i.e. 96 dense `[2560, 4, T]` broadcasts built and thrown
  away per forward pass at 48 layers x 2 sites, which is where a
  beat-llama.cpp-at-concurrency claim would come from. Not claimed here: no arm
  runs.

- **Nothing detects two claim files owning one matrix row**
  ([#2056](https://github.com/mudler/vllm.cpp/issues/2056)), and this row proved it
  rather than supposed it. W1 and W6a each wrote their own `CLAIM-*` for this row,
  both correct in isolation because each wave was the first product code from its
  own vantage. Copying one beside the other and running the checker gives
  `agent record OK ... rc=0`: git cannot conflict on it because the two sides touch
  different PATHS, and no gate reads for duplicate ownership. The collision was
  resolved by MERGE ORDER — W6a landed the row-level claim, W1 dropped its
  `-W1` file and deferred — which is an operator remembering, not a gate. Filed
  rather than fixed in flow because it changes checker semantics and so owes its
  own row, spec and red-before test per AGENTS.md §"Changing the rules or a
  checker". Owned by `MODEL-MM-QWEN4-EXP` until re-homed.

- **W5a (#2031) lands the GGUF WEIGHT LOADER, and it lands REACHED.**
  `src/vllm/model_executor/models/qwen4_exp_weights.{h,cpp}` materialize the
  text tower from a `qwen4exp` file, and `LoadQwen4ExpForConditionalGeneration`
  — the registry's `load_weights` hook, which a `qwen4exp` file already reaches
  through the `kGgufArchArms` dispatch row W6a added — calls it instead of
  refusing. This is the first slice of this row with a production call site.
  What it does NOT do is make the architecture SERVE: the forward and the
  KV-cache spec still refuse by name, so nothing decodes a token. Those two are
  W5b and W5c below.
- **W5b-1 (#2110) lands UNREACHED, by AGENTS.md "Nothing lands dead".**
  `RunGdnBlockPaged` and `BuildGdnStepInputs`
  (`include/vllm/model_executor/models/qwen3_5_gdn_block.h`, implemented beside
  `RunMoeBlock` in `src/vllm/model_executor/models/qwen3_5.cpp`) expose the
  qwen3_5 Gated DeltaNet block cross-TU so the `qwen4_exp` forward runs the SAME
  block its 36 `kLinearAttention` layers are, instead of growing a second copy
  of it. No production entry point calls either one at that commit: the
  Qwen3.5/3.6 forward keeps calling the anonymous-namespace `GdnBlockPaged`
  directly with its own `Dev` and its own step inputs, which is what keeps that
  path byte-identical, and the only caller of the public pair is the seam case in
  `tests/vllm/models/test_qwen3_5_gdn_spec_routing.cpp`. Row
  `MODEL-MM-QWEN4-EXP` owns the wiring, and W5b —
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), under
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978) — is the wave that
  composes it into `Qwen4ExpTextModel::Forward`. `RunMoeBlock` landed the same
  way and for the same architecture. This entry also discharges the FIRST
  sub-bullet of the W5b entry below for two of the seven names it lists: after
  #2110, `GdnBlockPaged` and (since `f730eb11c`) `MoeBlock` are both reachable
  from another translation unit. `FullAttnBlockPaged` and `RunLayerPaged` stay
  sealed and are not needed — this architecture has no full-attention layer and
  its own layer shape — and `StepDevInputs` / `BuildStepDevInputs` stay sealed on
  purpose, reached through the opaque `GdnStepInputs` handle.
- **A NUMERIC gate on the seam's `dh_fp8` argument needs a CUDA host.** The fp8
  W8A8 GDN input-projection tower is CUDA-only — `MatmulFp8CutlassD` and
  `MatmulFp8CutlassPreQuantD` both refuse unless `kMatmulFp8CublasLt` is
  registered — so on CPU the argument is gated by WHICH of those two leaves
  refuses, in both directions (`## Mutation record — W5b-1`). That is a genuine
  discriminator and it closes mutation C, but it compares no number. On a device
  that supports fp8 the same sub-case can compare the forwarded arm's output
  against `QuantFp8Static(h, input_scale)` fed through the plain arm, which is
  what the two leaves are documented to make identical. Owned by row
  `MODEL-MM-QWEN4-EXP` under
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978), which is where the
  row's other device-arm debts sit. #2110 closes with this change and so
  cannot carry it.
- **W5b, the forward, is OWED and it is the row's remaining barrier.** The
  scope is `Qwen4ExpTextModel::Forward` over 48 layers in `vt::` ops — the
  10240-wide hyper-connection stream, 36 Gated DeltaNet layers, 12 QSA layers,
  the 512-expert MoE with its shared expert, the PLE layer on 0-based layer 1,
  and the `use_combine=false` mixer that collapses the stream at the end. Two
  structural facts about this tree shape it, both MEASURED during W5a rather
  than assumed, and neither was in this spec before:
    * **`qwen3_5.cpp` lines 1209-7890 are one anonymous namespace.**
      `GdnBlockPaged`, `FullAttnBlockPaged`, `MoeBlock`, `SharedExpert`,
      `RunLayerPaged`, `StepDevInputs` and `BuildStepDevInputs` all have
      INTERNAL LINKAGE, so no new translation unit can call any of them. Reuse
      needs them hoisted into a header the way `dense_attn_block.h` was hoisted
      out of `qwen3.cpp` — a documented in-tree precedent, and an edit to a
      1745-line-plus file several other rows are working in. That extraction is
      its own unit of work and should be its own row.
    * **What IS free is the `vt::` op layer**, and it is enough to build the
      forward from: every `vt::Gdn*` entry point has a registered CPU kernel as
      well as a CUDA one, `vt::Moe*`, `vt::FusedChain`, `MRotaryEmbedding`,
      `dense_attn::AttnBlock` and `layers::MlpGateUpMethodBase` are all
      header-inline or externally linked. `muse_glimmer.cpp` builds a complete
      forward that way in 510 lines and is the shape to follow.
  W2/W3/W4 remain host-float references with `std::vector<float>` signatures;
  the forward needs their arithmetic in `vt::` ops, which is the "device arm"
  each of those waves already records as owed. Writing a host-float forward
  instead would be the hand-written parallel path AGENTS.md §"Shared seams"
  forbids, and it is recorded here so the shortcut is refused deliberately
  rather than rediscovered.
- **W5b-2 (#2123) lands the gated-residual DEVICE ARM, and it lands UNREACHED.**
  `vt::Qwen4ExpGatedResidual` and `vt::Qwen4ExpGatedResidualWriteBack`
  (`include/vt/ops.h`, dispatchers in `src/vt/ops.cpp`, CPU kernels in
  `src/vt/cpu/cpu_qwen4_exp.cpp`) are the hyper-connection stream in `vt::`
  ops, batched over T tokens. Nothing calls them from a production entry point
  at their merge commit: `Qwen4ExpTextModel::Forward` does not exist, and
  `ForwardQwen4ExpForConditionalGeneration` still refuses by name. The wiring is
  owed by **W5b, the forward**, under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), and the campaign row
  `MODEL-MM-QWEN4-EXP` tracks it under
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978). That is the same
  arrangement W2 (#1987), W3 (#1988) and W4 (#1991) landed under, and for the
  same reason: the only production entry point this architecture has is
  `ModelRegistry::Forward`, which is all-or-nothing, so every slice below the
  whole forward is unreached by construction.

  **Why an op and not a composition**, recorded because the alternative is the
  first thing a reviewer should ask about. Surveyed at `331eda888`: there is no
  ungated per-group RMS norm (`vt::RmsNormGated` has no `group_size`,
  `vt::RmsNormGatedGroup` requires a non-nullable SILU gate), `vt::RmsNorm`
  cannot carry a per-group weight (one `[H]` gamma per row against `hc_norm`'s
  `[hc*H]`), and there is no standalone `silu`, no standalone `sigmoid`, no
  elementwise binary multiply and no axis reduction anywhere in the op set —
  `kSilu`/`kSigmoid`/`kMul` exist only as `FOp` opcodes inside a `constexpr
  FusedRecipe` and are unreachable as free functions. A composition would need
  five new general ops and would still materialise the `[T, hc, H]` broadcast the
  rank-1 write-back exists to avoid. `kDeepseekV4Mhc` is the in-tree precedent.

  **Still owed from this wave, in order:**
  * **The CUDA arm.** Nothing is registered for any device but `kCPU`, so the
    dispatcher refuses by name everywhere else rather than falling back. It was
    not written because it could not be gated: this wave ran on a CPU host with
    no lease, and an ungated kernel is worse than an absent one.
  * **The reduction width that CUDA arm has to choose.** The per-group sum of
    squares accumulates in `double` here, and the W5b-2 table measures a 571x
    separation from a `float` accumulator at group size 2560. A straight
    f32-accumulate block reduction will not meet `kAccumBound`, so the device
    kernel must accumulate wider than f32 or be gated against the oracle
    directly. That is the same decision `## Owed` already records for the W3
    reference, arriving now with a number attached.
  * **One deliberate divergence from upstream, in the bf16 arm.**
    `Qwen4ExpTextRMSNorm.forward` is `self._norm(x.float()) * (1.0 + weight.float())`
    followed by `.type_as(x)`, so on a bf16 stream upstream ROUNDS the normed
    value before the down projection and before the `mixed_input` product. This
    op does not: it widens on load, computes in f32 and rounds once on the store,
    which is this tree's house contract and what `vt::RmsNorm` says of itself in
    the same terms. The f32 arm the goldens are dumped at is unaffected. What
    this costs is that a bf16 parity comparison against a running oracle carries
    a bf16-eps term an f32 one does not, and mutation M13 shows the two are
    distinguishable (reproducing upstream's rounding reds 8 assertions). Owed:
    that term stated in whatever first compares a bf16 arm to the oracle.
  * **A fused single-pass kernel.** The CPU kernel walks the stream four times
    per token (norm, down, up, collapse) plus once per injection row. That is
    correct and slow, and it is deliberate at a wave whose `## Gates` admit no
    speed number; the G4 axis is where it becomes a question.
  * **`hc_norm`'s provenance check.** The op takes vLLM's `1 + w_hf`
    parameterization and never adds 1. `vllm::qwen4_exp::HcNormWeightFromHf` is
    the one home of the fold and a `qwen4exp` GGUF carries it already, so the
    loader owes the cheap check this spec already describes — an unfolded gamma
    is zero-centred and a folded one is centred on 1.0 — at the point where the
    forward binds weights to this op.

- **The epsilon placement was ungated on the DEVICE arm, and W5b-2 closed it
  there and strengthened the host arm in the same flow.** Found while mutating
  the device kernel: moving `+ eps` outside the rsqrt SURVIVED every GOLDEN case
  on both arms, because A, B and C all draw the stream at `hyper_scale = 1.7`,
  where the mean square is O(1) and `eps = 1e-6` moves the answer by about a
  third of `kTol`; `Variant` in `test_qwen4_exp_hc.cpp` has no flag for it
  either. **W3's host suite nonetheless already gated the placement**, through a
  deliberate large-eps probe it shipped for exactly this reason
  (`test_qwen4_exp_hc.cpp:268-276`, whose own comment says a case at an eps large
  enough to separate the two spellings "is the only thing that gates it"): it
  compares `GroupedRmsNorm(..., 4.0f)` against `NormRefD(..., 4.0, Variant{})`,
  and at `big_eps = 4.0` the two spellings are 0.802185 apart against a `kTol` of
  1e-5. MEASURED on a reconstructed pre-repair tree (W3 host suite and goldens at
  `origin/main`, no case D) with the eps mutation applied to
  `qwen4_exp_hc.cpp`: **RED, 2 of 14 cases, 2 assertions** — the `big_eps` probe
  at :276 and golden case B at :343, whose own `eps = 1e-5` puts it marginally
  over tolerance. So the hole was the device arm's alone. Case D closes it there
  and additionally sharpens the host arm from 2 red assertions to the 10 M16
  records at head, which is an enhancement rather than a hole closed. The fourth
  golden case is drawn at `hyper_scale = 0.01`, generated from the same
  lane-pinned oracle source by the same script, and driven by BOTH suites; A, B
  and C regenerate byte-identically. Mutations M7 (device) and M16 (host) are the
  red-after measurement. The defect itself was fixed in the flow that found it,
  which is what AGENTS.md asks for a small and clear fix, so its record is this
  entry and the W5b-2 mutation table. One residual outlived that flow: the
  `#2123` row in `.agents/issue-index.md` still states that the placement
  survived in the W3 host suite, and a row there can never be edited because
  `merge=union` duplicates an edited row instead of merging it.
  [#2141](https://github.com/mudler/vllm.cpp/issues/2141) tracks that residual
  and is the key of the appended row that supersedes the claim; it needed an
  issue number of its own because `check-agent-record.py` refuses a second row
  keyed on `#2123`, reporting it as the duplicate two branches appending the
  same issue would produce.

- **W5c-1, the KV-cache spec, has LANDED**, and this bullet is what it
  replaces rather than a claim it is still owed. It publishes three groups (see
  `### The KV-cache spec is THREE groups and ONE uniform recurrent group`) and
  `MakeQwen4ExpKVCache` returns instead of refusing. The engine blocker this
  bullet named — the runner's `shapes.size() == 2` refusal — was closed by
  `ENG-RECURRENT-MULTISTATE`
  ([#2131](https://github.com/mudler/vllm.cpp/issues/2131), `f7710c1b4`), and
  the SECOND half it named, more than one recurrent group, turned out **not to
  be on this row's path at all**: upstream declares one recurrent shape
  model-wide, so `qwen4_exp` publishes ONE uniform recurrent group and a scalar
  `gdn_group_id_` carries it. The naming correction this bullet recorded is now
  fixed in the source it was about, in flow, as
  [#2198](https://github.com/mudler/vllm.cpp/issues/2198): W4's two comments in
  `src/vllm/model_executor/models/qwen4_exp_qsa.h` cited a `tokens_per_state`
  field with ZERO hits over the pinned vLLM tree and anchored it at an
  unrelated function; they now cite `compress_ratio`
  (`vllm/v1/kv_cache_interface.py:386`, `:393-395`, `:424-435`). The LOCAL
  `QsaSideCacheSpec::tokens_per_state` keeps its name — its arithmetic is right
  and pinned — with a comment saying it has no upstream referent.
- **THE N-GRAM HISTORY IS ZERO-SEEDED, AND 0 IS A VALID TOKEN ID. Nothing in
  this tree can see it.** `CacheBuffer` zero-fills every recurrent state it
  allocates (`src/vllm/v1/worker/gpu/runner.cpp`, both the host and the
  device-`Memset` arm), which is correct for every float state — zero bytes are
  `+0.0f` — and WRONG for the n-gram token history. Upstream's own
  `update_conv_state` pads with 0 too, and the model works around it with an
  explicit EOS left-pad; `PleSequenceState::Reset`
  (`src/vllm/model_executor/models/qwen4_exp_ple.h`) says so in terms: "Pad with
  EOS, never with zero." The forward must therefore EOS-seed that row on the
  same `prefill_has_initial_state == 0` predicate the GDN temporal state already
  uses (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1513-1514`,
  mirrored at `include/vllm/model_executor/models/qwen3_5.h`). W5c-1 publishes
  the state and CANNOT seed it, because there is no `Qwen4ExpTextModel::Forward`
  to seed it in — that is W5b. **No gate here can catch a zero seed**: token id
  0 hashes to a valid table row, so the model produces fluent wrong text, and
  the only oracle that would catch it is a transformers run this row cannot
  stand up (`gateable = no`).

  **W5e-2 ([#2336](https://github.com/mudler/vllm.cpp/issues/2336)) SEEDS IT,
  and the residual is narrower rather than gone.** `RunQwen4ExpPleBlock` seeds
  `caches.tokens` with `eos_token_id` and zeroes the conv ring on
  `past_len == 0`, which is `PleSequenceState::Reset` and is exactly upstream's
  `has_previous_state(layer_idx, state_idx=2) == False` branch. The claim that
  "no gate here can catch a zero seed" is FALSIFIED by that wave and the
  falsification is executable: spec mutation M2 replaces the EOS seed with zero
  and reds 4 of 11 cases, because the lane-pinned end-to-end golden
  `kPleExpectedOutput` was captured under upstream's EOS pad. The separation is
  measured at 1.2892 against a 1e-5 tolerance. What made the old sentence true
  was that no `qwen4_exp` suite ran the seeding and the hash and the gather
  together; one does now.

  **WHAT IS STILL OWED IS THE HOP THE BLOCK CANNOT TAKE.** The seeding is
  correct in the block and the block has no production caller, so on the path a
  runner actually drives, the zero-filled `CacheBuffer` row is still what a
  forward would read — because there is no forward. W5f, the layer loop, has to
  route the runner's own recurrent slot into `Qwen4ExpPleCaches::tokens` and pass
  the runner's `past_len`; passing a fresh scratch each step would seed
  correctly and lose the history. Owned by W5f under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031).
- **CLOSED by W5c-2 ([#2249](https://github.com/mudler/vllm.cpp/issues/2249)
  item 3): group 2's block table is gathered. What replaces it is NARROWER and
  it is still `## Owed`: NOTHING READS IT.** The entry this replaces said
  `GPUModelRunner::gather_block_table` is called for `full_attn_group_id_` and
  `gdn_group_id_` and for no other group, so the indexer group's per-request
  block rows never reach a forward. `GPUModelRunner::gather_group_block_tables`
  now gathers EVERY published group's table on the multi-cache path and
  publishes them by GROUP ID on `MultiKvCacheIndex`, mirroring upstream's
  per-group metadata loop (`vllm/v1/worker/gpu_model_runner.py:2551-2567` @ pin
  `5559679229`, `cm.block_table_tensor = _get_block_table(kv_cache_gid)`), so
  the map from a logical position to the physical page now reaches the forward.
  The half that remained was the CONSUMER, and **W5i closed it**:
  `Qwen4ExpQsaPagedCaches::index_key` was a contiguous
  `[max_kv, indexer_head_dim]` tensor — W5d-3 paged the K/V half of the axis and
  deliberately did not page the side cache — and it is now the engine's fused MLA
  page, stored and read through a block table. What is STILL owed is one hop
  further out: the map W5c-2 gathers reaches `MultiKvCacheIndex` and not the
  block, because `ModelRegistry::Forward` refuses `multi_kv` (#2353) and the
  registry hook substitutes a per-call scratch in the same paged shape. **W5j
  owns that hop.** Named under all four "Nothing lands dead"
  conditions: what is unreached is the per-group block-table channel's VALUE —
  the runner's gather runs on the production `execute_model` path and the
  refusal in `ModelRegistry::Forward` reads its count, but no forward consumes
  the tables, because `ForwardQwen4ExpForConditionalGeneration` refuses by name;
  the row that owns the wiring is `MODEL-MM-QWEN4-EXP` at **W5** (the layer loop
  itself, which is what the W5a-W5d prerequisite waves feed; W5b-1..6 are all on
  `main`, so naming W5b here would send the reader to finished waves); the
  issues that track it are
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) and
  [#2249](https://github.com/mudler/vllm.cpp/issues/2249); and it is listed
  here, which is the `## Owed` entry the rule requires. The buffer itself IS
  allocated and gated — `test_runner.cpp`'s
  "a multi-cache topology ALLOCATES its N-state recurrent group" asserts the
  per-group pages — so this is an unread cache and not an unallocated one.
- **EVERY BYTE FIGURE IN THIS ROW IS DERIVED, NOT MEASURED.** On a CPU host
  `kv_cache_backend_resident_` is false
  (`!platforms::GetPlatform(dev.type).is_cpu()`), so the runner takes host
  vectors and nothing on a device has ever held this model's KV. The 3391504 B
  page, the 49.2 MiB uniform slack and the 256 B/token/layer side cache are
  arithmetic over the published shapes, gated as literals, and they are not a
  measurement. The side-cache figure read 64 B until W5h, which is the same
  arithmetic over a `compress_ratio` the oracle says this cache does not have. Gateable only on `dgx:gpu0`, and `--device cuda` still refuses
  ahead of any tensor I/O for the n-gram expansion
  ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)).
- **`--kv-cache-dtype fp8` now refuses the WHOLE model**, and that is a
  consequence of publishing an MLA group rather than a defect of it.
  `ApplyCacheDType` refuses any `MLAAttentionSpec`
  (`src/vllm/v1/kv_cache_interface.cpp`, `RetypeAttentionSpec`) because upstream
  gives an MLA page its own quantized formula (`fp8_ds_mla`,
  `kv_cache_interface.py:398-410`) and this tree has the formula with no
  fp8_ds_mla store or read. Gated as an executable consequence in
  `test_qwen4_exp_kv_cache.cpp` rather than left to be discovered from a command
  line. The fp8_ds_mla read/write side is NOT this row's; `auto` is unaffected
  and is the production default.
- **The >2048-token QSA gate still has no forward to run.** `## Gates` requires
  a QSA correctness gate past `indexer_budget` tokens of context, because below
  it every candidate block is selected and a pooled-key defect is invisible.
  W5c-1 publishes the cache that gate needs and decodes nothing. Owed by W5b.
- **The VISION path is owed and has no GGUF artifact to load.** The tower is an
  unchanged `Qwen3_5MoeVisionModel`, but the shipped `unsloth` UD-IQ1_S file is
  TEXT-ONLY: its 1224 tensors are 768 hyper-connection/MoE, 324 Gated DeltaNet,
  120 QSA, 6 PLE and 6 model-level, and there is not one `v.blk.*` or `mm.*`
  among them. So the multimodal arm needs either a companion mmproj that does
  not exist yet or the safetensors arm that no device we own can hold. Recorded
  as a BLOCKER rather than as scheduling.
- **The safetensors arm is refused for a reason, and the refusal now says it.**
  Every published safetensors artifact — bf16 ~360 GB, FP8 ~180 GB, NVFP4
  ~128 GB — exceeds every device this project owns, so an arm that read them
  would be code nothing could run. W5a rewrote the message from "not ported
  yet", which reads as scheduling, to the size argument plus the name of the
  arm that IS supported.
- **The converter and the algorithm oracle DISAGREE on which EOS the n-gram
  hash uses, and a GGUF-only load has to take the converter's.** llama.cpp
  #27742 resolves `qwen4exp.ple.eos_token_id` as `int(eos[-1])`, the LAST
  element of the HF list; `Qwen4ExpTextModel.forward` takes element `[0]`. On a
  single-entry list they coincide and nothing shows. On a longer one they
  disagree, and the disagreement is invisible because both runtimes emit fluent
  text from different n-gram segment boundaries. `ParseQwen4ExpParams` follows
  the algorithm oracle wherever a `config.json` is present; from a GGUF the
  container is the only source. **LATENT on this checkpoint, not active:** the
  released `Qwen/Qwen3.8-Flash-Next` `config.json` carries
  `eos_token_id: 248044` as a bare integer, so `[0]` and `[-1]` are the same
  value and the two runtimes agree today. Owed: the same check on any future
  checkpoint of this family whose `eos_token_id` is a list.
- **The V-head reorder costs the Gated DeltaNet tower its keep-quant
  residency. MEASURED: net +2.446 GiB, and NOT fit-threatening.**
  `_LinearAttentionVReorderBase` fires whenever `num_k_heads != num_v_heads`,
  which is 16 vs 48 here, so every GDN projection of all 36 linear layers is
  layout-rewritten at load and therefore `kTransformedWeight` — it expands to
  bf16 instead of staying Q5_K/Q6_K. The ROW reorders could in principle be done
  inside the block stream (a k-quant row is a whole number of superblocks, so
  moving whole rows never cuts one), but `out_proj`'s is a COLUMN permutation
  and can never be. `qwen3_5_gguf_weights.cpp` already has this property for the
  27B.

  The earlier version of this entry said "resident-bytes unmeasured, because no
  real file has been loaded". **No file is needed.** The committed 1224-tensor
  manifest (`tests/vllm/models/qwen4_exp_gguf_manifest.inc`,
  `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` @
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`) carries every name, `ne` and ggml
  type id, and the block geometries are `src/vt/dtype.cpp`'s own table (Q5_K
  256/176, Q6_K 256/210, F32 1/4). The whole file sums to 72,535,436,800 B =
  **67.554 GiB**, which agrees with the 67.56 GiB this spec states from the
  repository listing — so the arithmetic below is cross-checked against a number
  measured a different way.

  The five tensors the reorder moves off the keep-quant arm, over 36 GDN layers:

  | tensor | ggml type | on disk | resident bf16 |
  |---|---|---|---|
  | `attn_qkv.weight` [10240, 2560] | Q5_K | 652,288,000 B (0.6075 GiB) | 1,887,436,800 B (1.7578 GiB) |
  | `attn_gate.weight` [6144, 2560] | Q5_K | 391,372,800 B (0.3645 GiB) | 1,132,462,080 B (1.0547 GiB) |
  | `ssm_out.weight` [2560, 6144] | Q6_K | 464,486,400 B (0.4326 GiB) | 1,132,462,080 B (1.0547 GiB) |
  | `ssm_alpha.weight` [48, 2560] | F32 | 17,694,720 B (0.0165 GiB) | 8,847,360 B (0.0082 GiB) |
  | `ssm_beta.weight` [48, 2560] | F32 | 17,694,720 B (0.0165 GiB) | 8,847,360 B (0.0082 GiB) |
  | **total** | | **1,543,536,640 B (1.4375 GiB)** | **4,170,055,680 B (3.8837 GiB)** |

  **Net +2,626,519,040 B = +2.446 GiB.** The two `ssm_*` rows NARROW, because
  the file stores them F32 and we hold them bf16; only the three quantized
  projections grow.

  `ssm_conv1d.weight` is deliberately NOT in that table even though it also
  lands bf16 (5,898,240 B -> 2,949,120 B over 36 layers). It is a depthwise
  filter that `LoadGdn` dequantizes whether or not the reorder fires, so its
  residency is not a cost of the reorder. Including it would move the "on disk"
  total to 1.4430 GiB and the net to +2.443 GiB, which is the difference between
  this figure and a first reading of it.

  **Not fit-threatening.** 67.554 GiB on disk against ~119.6 GiB usable on GB10
  leaves ~52 GiB, and +2.446 GiB is 4.7% of that headroom. The residency
  question that DOES threaten the fit is the n-gram gather table on a non-CPU
  device, which is +68.5 GiB and is the next entry.
- **The CUDA arm cannot gather from the n-gram table, so it is REFUSED at load
  ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)).**
  `DeviceQuantGatherSupported` is true for `kCPU` alone, so on any other device
  `RouteGgufTensor` sends `per_layer_token_embd.weight` to `kExpandBf16`. From
  the same manifest that tensor is [320001536, 160] IQ4_NL: **26.822 GiB on
  disk, 95.368 GiB expanded** (320001536 x 160 x 2 = 102,400,491,520 B), on a
  box with ~119.6 GiB for everything. The #1123 device-fit guard sums the file's
  ON-DISK bytes — 67.554 GiB, comfortably inside the budget — so it admits the
  load and the expansion happens after it, which is `model_loader.cpp`'s own
  stated worst case: "Loading for 26 minutes and dying mid-stream is the worst
  of the available behaviours."

  W5a's repair takes the device as an argument to `LoadQwen4ExpFromGguf` — no
  default, so a caller cannot disable the guard by saying nothing — and refuses
  BY NAME ahead of any tensor I/O.

  **KGATHER WROTE THAT KERNEL AND FLIPPED THE GATE**
  ([`cuda-quant-gather.md`](cuda-quant-gather.md)), so the sentence above —
  "`DeviceQuantGatherSupported` is true for `kCPU` alone" — is now true only of
  METAL, VULKAN, ROCM and TENSTORRENT. `EmbeddingKernelCuda` decodes a block row
  across all 18 encodings `vt::cpu::BlockToFloat` decodes, and the gate is no
  longer a device list at all: `vt::Embedding` routes a block table to
  `OpId::kEmbeddingQuant` and `DeviceQuantGatherSupported` is
  `OpRegistered(kEmbeddingQuant, dev)`, so a backend advertises the capability by
  registering the kernel. MEASURED on `thor:gpu0` (sm_110, nvcc 13.0.88,
  2026-08-31, job `e53a20f5`): 6 of 6 cases at 231 assertions, every encoding
  bit-exact against the CPU arm, against a pre-arm RED leg that failed by name at
  32 assertions. The guard itself is UNCHANGED and still refuses the other four
  devices by name.

  **The n-gram table therefore stays block-resident on a CUDA card**, 26.822 GiB
  instead of 95.368 GiB, which is what the load-side blocker was. It does NOT
  make this model run on a GPU: `ModelRegistry::Forward` is all-or-nothing and
  several `qwen4_exp` ops have no CUDA arm, so `--device cpu` is still the answer
  for a token. **What KGATHER does NOT do:** it removes the
  LOAD-side blocker and produces no token on a GPU, because no `qwen4_exp` op
  reaches a CUDA queue (W6-CUDA landed four arms; `vt::RmsNormGroup`, the QSA
  block and the rest are still host-only, and `ModelRegistry::Forward` is
  all-or-nothing). It claims no throughput number.

  **Still owed after KGATHER, and named rather than implied:**
  - **sm_121a, and M2 on a device** —
    [#2393](https://github.com/mudler/vllm.cpp/issues/2393). The device gate ran
    on sm_110 (thor). A `dgx:gpu0` job is queued to confirm the target
    architecture and to close the one mutation that did not build there (M2, an
    `-Werror` artifact on an unreferenced local, now fixed). No sm_121a claim is
    made.
  - The gather arms of METAL, VULKAN, ROCM and TENSTORRENT —
    [#2394](https://github.com/mudler/vllm.cpp/issues/2394).
  - Any performance claim. The decoders read byte-wise for alignment safety and
    one thread decodes a whole block; no benchmark ID moves.

  **Also owed, and only NARROWED by that guard: the load still runs to
  completion on `--device cpu` and then dies in `MakeQwen4ExpKVCache`.** Before
  W5a the loader refused at once; after it, a CPU user pays the full load first.
  W5c closes this by making that function return a config rather than throw.
- **`ReorderVRows`/`ReorderVCols` exist twice in this tree, and only ONE copy is
  gated.** `qwen3_5_gguf_weights.cpp` has them in an anonymous namespace with no
  header, and `qwen4_exp_weights.cpp` has its own copy. Four lines of index
  arithmetic, deliberately duplicated rather than hoisted: the hoist edits a
  1745-line translation unit other rows are working in, which is the same
  shared-file lock the `qwen3_5.cpp` extraction above runs into. Owed to
  whichever row does that extraction.

  **The earlier version of this entry said "gated on both sides", and that was
  FALSE.** Measured at `a68312c79` by the W5a repair: swapping `g` and `t` in
  the `qwen4_exp_weights.cpp` copy reddens `test_qwen4_exp_gguf_weights` (2
  cases, 41 assertions, mutation M5), while the same swap in the
  `qwen3_5_gguf_weights.cpp` copy leaves `test_gguf_qwen36_loader` (7/7, 555
  assertions),
  `test_model_loader_gguf` (7/7), `test_gguf_nvfp4` (14/14) and
  `test_gguf_keep_quant` (42/42) ALL green — every synthetic `qwen35`/
  `qwen35moe` fixture in the tree is `ssm.group_count = 2,
  ssm.time_step_rank = 4`, i.e. K == R, where the permutation is its own
  inverse. The qwen3_5 side is tracked by
  [#2081](https://github.com/mudler/vllm.cpp/issues/2081) and is deliberately
  NOT fixed under this row: re-shaping a shipped model's fixtures changes
  `qwen35`, `qwen35moe` and `qwen3next` coverage and is not this row's scope.

  **#2081 now has an `issue-index.md` row, and it did not before.** The issue was
  open on GitHub and named here, but the index carried nothing for it, so two of
  the three places AGENTS.md requires to agree did not. The appended row names
  `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation` as the owner — the
  shipped model whose fixtures have to change — and points back at this `## Owed`
  entry. Two corrections ride with it, both from re-measuring MUT-M6 on this
  branch rather than relaying the earlier numbers. The survival holds exactly
  (7/7 555, 7/7 23, 14/14 2352, 42/42 6340, every count identical to the
  un-mutated baseline). The stated CAUSE was too narrow: the fixtures are not all
  `ssm.time_step_rank = 4`. `test_gguf_qwen36_loader`'s default shape is
  `group_count = 2, time_step_rank = 2`, i.e. R = 1, where the map is the
  IDENTITY and no inversion is even expressible; only the one case named "V-head
  reorder when num_v != num_k" reaches R = 2, and that is the self-inverse
  K == R. Both roads end at the same place, but a reader chasing "K == R" through
  the default fixture would not find it.
- **W5b-6 (#2218) RESOLVES THE GAMMA POLARITY, AND IT RESOLVES IT THE OTHER WAY
  ROUND FROM WHAT THAT ISSUE PROPOSED.** #2218 asked the layer loop to fold
  `hc_norm`, `norm_key`, `norm_query` and `norm_conv` through
  `vllm::qwen4_exp::HcNormWeightFromHf` before use. Folding the last three would
  have been the same defect moved one tensor to the left: their consumers
  already add the 1. Measured in this tree rather than argued —
  `RunQwen4ExpQsaBlock` normalizes THREE of its four QSA gammas —
  `idx_q_norm`, `q_norm` and `k_norm` — through `vt::RmsNorm(..., gemma = true)`,
  which is `out * (1 + w)` (`qwen4_exp_qsa_block.cpp:383-384`, `:425-426`,
  `:441-442`, three line pairs for three gammas). **The fourth, `idx_k_norm`,
  never reaches `vt::RmsNorm` at all**: it is handed to `Qwen4ExpQsaIndex`
  (`:401-403`) and consumed inside it by `vt::Qwen4ExpQsaCompress` (`:181`),
  which documents `k_norm_weight` as "the HuggingFace gamma, applied as
  `(1.0 + weight)` ... NOT vLLM's `out * weight`". The polarity is the same
  either way, which is why the conclusion below is unaffected, but the CONSUMER
  is a different op and this entry said `vt::RmsNorm` for all four until the
  W5b-6 review counted the citations against the claim. The PLE host reference
  spells `(1.0 + static_cast<double>(weight[base + i]))` inline at
  `qwen4_exp_ple.cpp:72`. **Three of the four consumers were already on the
  loader's convention and only `vt::Qwen4ExpGatedResidual` was not**, so the op
  moved rather than the loader. The rule is now one line: every gamma in
  `Qwen4ExpWeights` is the RAW HuggingFace parameter and every consumer adds the
  1, `linear_attn.norm.weight` excepted because the converter never folds it and
  `vt::RmsNormGated` wants a plain multiplier. That is also upstream verbatim,
  `Qwen4ExpTextRMSNorm.forward` = `output * (1.0 + self.weight.float())` over a
  zero-initialised parameter (`modeling_qwen4_exp.py:173-178`).
  `HcNormWeightFromHf` survives as the bridge to the W3 HOST reference, whose
  `GroupedRmsNorm` keeps vLLM's `out * w` form, and it is now called from the
  two suites that drive that reference and from no production path.
- **W5b-6 (#2218) LANDS UNREACHED, by AGENTS.md "Nothing lands dead".**
  `vt::Qwen4ExpGatedResidual` and `vt::Qwen4ExpGatedResidualWriteBack`
  (`include/vt/ops.h`, dispatchers `src/vt/ops.cpp`, CPU kernels
  `src/vt/cpu/cpu_qwen4_exp.cpp`) are the ops whose gamma contract this wave
  changed, and at its merge commit nothing calls either from a production entry
  point. Their only call sites are `tests/vllm/models/test_qwen4_exp_hc_device.cpp`
  and the new `tests/vllm/models/test_qwen4_exp_forward.cpp`. That second suite
  reaches the PRODUCTION LOADER — `ModelRegistry::Load` over a `qwen4exp` file —
  and it is what makes the fix gateable at all, but a test driving a production
  loader is still a test: it is not a production entry point, and reaching the
  loader does not reach the op. `Qwen4ExpTextModel::Forward` does not exist and
  `ForwardQwen4ExpForConditionalGeneration`
  (`src/vllm/model_executor/models/qwen4_exp_registry.cpp`) still refuses by name
  before any downcast, so the op stays unreached for exactly the reason W5b-2
  (#2123) recorded when it landed the op in the first place. The wiring is owed by
  **W5b, the layer loop**, under
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), owned by row
  `MODEL-MM-QWEN4-EXP` and tracked by campaign
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978); the five measured
  prerequisites that wave must clear first are the entry below this one.
- **THE LAYER LOOP'S PREMISE — "every component it composes is already on
  `main`" — IS FALSE, AND HERE ARE THE FIVE THINGS THAT ARE NOT.** Surveyed
  against this tree while W5b-6 was in flight, each independently sufficient to
  stop a token, and each now named in the `ForwardQwen4ExpForConditionalGeneration`
  refusal so the next reader does not re-derive it:
    1. **CLOSED by W5d-1 (#2249 item 1): there is no standalone grouped RMS
       norm — there is now.** The refusal string that still said otherwise is
       [#2288](https://github.com/mudler/vllm.cpp/issues/2288), fixed by
       [#2265](https://github.com/mudler/vllm.cpp/pull/2265). `vt::RmsNormGroup` / `OpId::kRmsNormGroup` is that
       op, registered on `kCPU`, gated by `tests/vt/test_ops_rms_norm_group.cpp`
       at 7 cases / 69 assertions, and its own mutation record is the
       `## Mutation record — W5d-1` section above. The survey text is kept below
       because it is the argument that produced the op and the layer loop still
       has to CALL it, which nothing does yet. `Qwen4ExpTextPLELayer` holds
       three `Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=hidden_size)` —
       `norm_key`, `norm_query`, `norm_conv` — reducing over `hc` independent
       `hidden_size` slices of the 10240-wide stream. `include/vt/ops.h:556`
       states the gap in its own words: "`kRmsNormGated` has no group_size;
       `kRmsNormGatedGroup` requires a SILU gate". `vt::RmsNorm` reduces the
       whole last dim and takes a `[D]` gamma, and the PLE gamma is `[hc*H]`, so
       it cannot express this even per-branch. The only grouped reduction in the
       tree is FUSED inside `vt::Qwen4ExpGatedResidual` and is not exposed.
       **This is new op work, and it is the same "why a fused family op" argument
       W5b-2 made, arriving at the opposite answer because PLE needs the norm
       without the mix.**
    2. **CLOSED by W5d-3 (#2249 item 2), for the K/V half only: the QSA
       consumer is CONTIGUOUS and the published cache is PAGED — it now reads
       the paged one.** `Qwen4ExpQsaPagedCaches` and `RunQwen4ExpQsaBlockPaged`
       bridge KV group 0 (the `FullAttentionSpec`) through a
       `kv_block_table`/`kv_block_size` address mode inside the same
       `vt::Qwen4ExpQsaGatherAttention`, rather than a second op. The INDEXER
       side cache was untouched and still contiguous, which is item 3 below and
       was owed as W5c-2 — so this item closed and item 3 did not, and the two
       are the SAME axis split in half. **W5i closed the other half**, and by a
       different shape: a composition of `vt::IndexCopy` and `vt::IndexSelect` in
       the block, not a second address mode on an op. The two halves therefore
       set no single precedent, and a reader who takes W5d-3's address mode as
       THE pattern will over-extend an op that did not need it. The survey text follows, because it is
       the argument that produced the wave and the layer loop still has to CALL
       the paged arm, which nothing does.
       `Qwen4ExpQsaCaches` is `key`/`value` `[max_kv, num_kv_heads, head_dim]`
       and `index_key` `[max_kv, indexer_head_dim]`
       (`qwen4_exp_qsa_block.h`), while `MakeQwen4ExpKVCache` publishes a
       `FullAttentionSpec` and an `MLAAttentionSpec` the runner allocates as
       paged `CacheBuffer`s. Bridging them is a paged arm of
       `RunQwen4ExpQsaBlock`, not a cast.
    3. **CLOSED by W5c-2 (#2249 item 3): group 2 was allocated and unread —
       its block table is now gathered.** This item said `gather_block_table`
       has three call sites and reaches exactly `full_attn_group_id_` and
       `gdn_group_id_`. It has a FOURTH now,
       `GPUModelRunner::gather_group_block_tables`, which runs over every
       published group on the multi-cache path and publishes the tables by
       group id on `MultiKvCacheIndex`. What the loop still needs from this
       axis is the CONSUMER: item 2's paged arm reads the K/V through a block
       table and the INDEXER side cache off a contiguous `[max_kv, D]` array,
       so the map this item delivers has no reader. That stays carried under
       `## Owed` above.
    4. **The MoE weights need an adapter.** `Qwen4ExpMoeWeights` holds stacked
       `gate_exps`/`up_exps` `[E, moe_I, H]` and `down_exps` `[E, H, moe_I]`;
       `RunMoeBlock` reads `MoeBlockWeights`, whose arms are per-expert
       `[H, I]` vectors, an `Nvfp4Weight` set, or the stacked keep-quant
       `expert_gate_kq [E*I, H]` / `expert_down_kq [E*H, I]`. **CLOSED by W5d-4
       ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4),
       `src/vllm/model_executor/models/qwen4_exp_moe.{h,cpp}`. The sentence that
       used to stand here — "the third arm's shapes are exactly the qwen4_exp
       ones", so the adapter is "a reinterpretation … rather than a copy" — is
       measured FALSE, and it is false on the arm every shipped checkpoint
       takes.** `LoadStackedExperts` records the tower as RANK 3 `[E, N, K]`
       (`qwen4_exp_weights.cpp:160-164`) and `MoeBlockWeights::expert_*_kq` is
       RANK 2 `[E*N, K]`; the default keep-quant route
       (`Qwen35GroupedMoeEnabled`, ON) hands that tensor to
       `vt::MatmulBTQuantGrouped`, whose first check is
       "matmul_bt_quant_grouped: rank-2 out/act/weight required"
       (`src/vt/ops.cpp:223`). **That refusal is ROUTE-CONDITIONAL and the
       sentence above is scoped to the default route on purpose.** With
       `VT_QWEN35_GROUPED_MOE=0` the seam takes the per-expert `ExpertMlpKq`
       path, which reaches `KqResidentSlice` (`qwen3_5.cpp:5665-5678`); that
       helper rebuilds a rank-2 view from its `N`/`K` ARGUMENTS by pointer
       arithmetic, sets `wt.rank = 2` itself and never reads the tower's
       declared rank, so a rank-3 tower does not throw there — and, the tower
       being contiguous `[E, N, K]`, `row_off = e * N` lands on exactly the
       right slice, so it answers correctly. #2249 item 4's original sentence is
       therefore literally true of the NON-default route. It is false of the one
       every shipped checkpoint takes, which is why this wave was the size it
       was. Three more differences a shape comparison cannot
       see: the router and shared gate are f32 by `LoadMoe`'s deliberate choice
       and the CUDA GEMM refuses a (bf16, f32) pair by name
       (`cuda_matmul.cu:397-403`), so passing them through runs on CPU and dies
       on every GPU; `MoeBlock` selects the whole expert path from
       `expert_gate_kq` ALONE, so a per-tensor residency split reads as
       keep-quant and dereferences an empty tower; and a **bf16** tower cannot
       use the stacked fields at all (`ops.cpp:231` refuses a non-block dtype),
       so the bf16 arm fills the PER-EXPERT vectors — with zero-copy borrowed
       views, because three copies per layer at the released geometry is 240 GB
       across the stack. Both arms are now gated against a from-scratch
       reimplementation of the lane-pinned oracle in
       `tests/vllm/models/test_qwen4_exp_moe.cpp`. **And the alternate route is a
       measured RESULT rather than an admitted limit.**
       `Qwen35GroupedMoeEnabled()` caches in a function-local `static const`
       (`qwen3_5.cpp:6299-6302`), which prevents flipping it MID-PROCESS, not
       before launch — so `VT_QWEN35_GROUPED_MOE=0` does exercise `ExpertMlpKq`.
       Run that way, both value cases pass at BIT-IDENTICAL `max|diff|` to the
       default route (bf16 `0.00218359`, keep-quant `0.00865547`), which is the
       seam's own byte-identity claim at `qwen3_5.cpp:7261` measured rather than
       inherited. The suite runs on both routes and says which behaviour it is
       asserting on each.
    5. **CLOSED by W5d-2 (#2249 item 5, `3ed2378a3`): the mRoPE table builder
       has internal linkage — it no longer does.** This half of
       [#2288](https://github.com/mudler/vllm.cpp/issues/2288) is pre-existing
       debt from [#2264](https://github.com/mudler/vllm.cpp/pull/2264), which
       corrected this prose and left the refusal string; both are fixed by
       [#2265](https://github.com/mudler/vllm.cpp/pull/2265). `BuildMropeCosSinHost` is
       declared at `include/vllm/model_executor/models/qwen3_5_mrope.h:55` and
       defined without `static` at `qwen3_5.cpp:9475` on this merged head, so
       the QSA half can build the SAME tables the Qwen3.5/3.6 VL drivers build.
       The call from the loop is still owed, and it is W5b's. The survey text
       follows.
       `BuildMropeCosSinHost` WAS `static` at `qwen3_5.cpp:9472`, and
       `RunQwen4ExpQsaBlock` needs BOTH layouts derived from it: the packed
       bf16 `[P, rot]` `cos|sin` cache `vt::RopeFromCache` reads and the two
       separate f32 `[P, rot]` tables `vt::Qwen4ExpQsaCompress` reads, with
       `CheckRopeLayoutsAgree` verifying they describe the same angles.
  **And one more that is not this row's:** `ModelRegistry::Forward` refuses ANY
  non-null `multi_kv` by name (`model_registry.cpp:462-478` — `:428-440` on
  W5c-2's parent; W5c-2's own `BlockTableForGroup` sits above it and moved it,
  and the refusal now also reports how many groups have a gathered table), and
  this model's three published groups make the runner set it
  (`runner.cpp:787-804`, `:2325`). A forward reached through
  `ModelRegistry::Forward` with a hand-built positional cache set is gateable
  today; a forward reached through `GPUModelRunner` is not, and lifting that
  refusal is an engine seam change DeepSeek-V4 waits on too.

### The llama.cpp G4 denominator, owed after the arm landed

The `llama-cpp-qwen4exp` oracle is `gateable = yes` and its harness is committed,
and three things are still owed before a single cell of that table exists.

- **[#2261](https://github.com/mudler/vllm.cpp/issues/2261): a
  `KV_BYTES_PER_TOKEN` measured on a leased load, without which the ladder
  refuses.** `llama-server` at the pin reports no KV size — measured on the row's
  own production capture, which is the complete unfiltered server output — so a
  guard that defaulted the term to zero was weightless on the only server this
  harness will face. It now refuses (`E_KV_UNREPORTED`, 21) rather than sizing a
  49,152-token context over 32 slots against a 67.5 GiB model with nothing
  bounding the cache. A refusal is the correct output until the number exists.
- **[#2262](https://github.com/mudler/vllm.cpp/issues/2262): the mutation sweep
  is not re-executable and the CUDA toolchain is asserted rather than pinned.**
  Nine of eleven recorded mutations have no committed driver, and
  `cuda-toolkit-13-0` names a channel rather than a version; `EXPECT_NVCC` makes
  the drift refuse but does not make apt serve one build.
- **The ladder run itself, and a vllm.cpp arm to put beside it.** There is
  nothing to compare until `ModelRegistry::Forward` stops refusing `qwen4_exp` by
  name, and the vllm.cpp cell must be a TEXT-ONLY configuration on the same
  UD-IQ1_S artifact, because the denominator reports every modality false.


- **`ple.layer_multipliers` IS WRITTEN INTO THE TEXT CONFIG AND READ BY NOTHING,
  exactly as `ple.head_offsets` was until W5g.** `Qwen4ExpHfConfigFromGguf` sets
  `text["ple_layer_multipliers"]` and no code path consumes it;
  `BuildNGramTableLayout` always derives the multipliers from a splitmix chain
  over `vocab_size`, `ngram_size`, the PLE layer index and `seed`, where `seed`
  is 1234 because the published `config.json` states none.

  **WHAT IS NOT AT RISK, corrected by the W5g review.** An earlier wording of
  this entry said the released artifact "has not been read for it" and that a
  converter disagreement would make "every n-gram row from a real file somebody
  else's". Both overstate it. `tests/vllm/models/qwen4_exp_ple_goldens.inc`
  records `kRealLayerMultipliers = {23703573157769, 20109073645365,
  8052911324071}` with the provenance "matches the three values published in
  issue #1987 and range-read from the released safetensors", and
  `test_qwen4_exp_ple.cpp` asserts our seed-1234 derivation against them
  element-wise. So for the artifact that matters our derivation is already
  verified against the shipped buffer, and the released file's rows are not in
  question.

  **WHAT IS STILL UNREAD is the GGUF CONTAINER KEY** — `qwen4exp.ple.layer_multipliers`
  as llama.cpp #27742's converter writes it, at its pin. The residual risk is
  narrower than the derivation being wrong: a file generated at a DIFFERENT seed,
  or a converter that writes multipliers disagreeing with the checkpoint's own
  buffer. Either would gather rows we could not detect, because we ignore the key
  and the mismatch produces no shape error. The synthetic fixture writes no such
  key, so nothing here observes it either way. NOT fixed in this wave because the
  repair needs the container oracle read at its pin, which is the same evidence
  W5g gathered for the head arrays and did not gather for this one.
  Owned by `MODEL-MM-QWEN4-EXP`; NO ISSUE NUMBER, GitHub writes are `403`.

- **A STATED HEAD-VOCABULARY SET IS REFUSED ON ANY PLE LAYER BUT INDEX 0.** W5g
  makes this explicit rather than silently wrong (see above). It is a real
  limitation of the container format as read at the pin: one flat array of
  `ngram_heads` entries, with nothing saying which PLE layer it describes, while
  upstream derives a different set per layer. The released checkpoint has one PLE
  layer so nothing published hits it. Closing it needs the converter re-read to
  learn whether a multi-PLE-layer file states a longer array or one per layer.
  Owned by `MODEL-MM-QWEN4-EXP`; NO ISSUE NUMBER, GitHub writes are `403`.

- **THE RETAINED CHAIN-VS-STATED CROSS-CHECK IS REACHABLE FROM NO SHIPPED
  SOURCE.** W5g narrowed the head-vocabulary cross-check in `Qwen4ExpPleLayout`
  to sources that STATE `ngram_vocab_size_base`, which is correct — the
  unnarrowed form compared a file against a default and refused correctly loaded
  weights. The consequence, which the W5g review established and this entry
  records rather than leaves implicit: the guard now needs a stated head-size set
  AND a stated base, and nothing shipped states both. A `qwen4exp` GGUF states
  the sizes and never the base (`Qwen4ExpHfConfigFromGguf` writes
  `ple_head_vocab_sizes` and no `ngram_vocab_size_base`); the released
  `config.json` states the base and never the sizes. The only thing that drives
  it on this head is `test_qwen4_exp_ple_block.cpp`'s `GoldenParams()`, which
  sets `ngram_vocab_size_base_stated` by hand. "Narrowed, kept" without this
  sentence implies a production arm that does not exist.

  It is KEPT rather than deleted because the case it protects is real and one
  converter commit away — a source that writes the resolved arrays AND the base
  it derived them from — and because the failure is silent: `head_offsets` is an
  exclusive prefix sum, so one wrong size re-points every later head at another
  head's rows with no shape error. Closing this means either reading llama.cpp
  #27742's converter at its pin to learn whether it can be made to state both,
  or removing the guard and saying what replaces it. Owned by
  `MODEL-MM-QWEN4-EXP`; NO ISSUE NUMBER, GitHub writes are `403`.

### TIEBREAK: two gaps the wave RECORDED rather than fixed (#2586)

Wave TIEBREAK's fresh review found two properties of
`tests/vt/test_moe_router_tie_stability.cpp` that are real and that this row
owes. Neither is fixed in the wave's own pull request: the first is a tree-wide
test idiom and the second needs GPU time to re-measure every count the wave's
records cite, so folding either in would make a test-only change carry an
unmeasured claim.

- **A DEVICE-LESS CUDA BUILD READS GREEN
  ([#2603](https://github.com/mudler/vllm.cpp/issues/2603)).** The three device
  cases guard on `if (!HasCuda()) { MESSAGE(...); return; }`, so a CUDA build on
  a host with no device reports **4 cases | 488 assertions | 0 failed | rc 0**
  while measuring nothing on device. The wave's own `orin:gpu0` lease did exactly
  that, green, with a broken tie-break in the binary (evidence §8). Three states
  exist and the assertion count separates only two of them — CPU-only is
  `1 | 488`, device-less CUDA is `4 | 488`, a real device run is `4 | 4652` — so
  the Gates section now teaches the CASE count beside it. The tree already has
  the refusing idiom at
  `tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:149`
  (`VT_REQUIRE_27N_FP8_GATE` turns absence into a FAILURE), and 13 other test
  files share the permissive shape, so adopting it is a row of its own.

- **HALF THE DEVICE SWEEP IS A DUPLICATE ABOVE E = 256
  ([#2604](https://github.com/mudler/vllm.cpp/issues/2604)).** Case 1 runs its
  sweep with `VT_MOE_ROUTER_WARP` pinned `"1"` and `"0"`, but
  `MoeRouterWarpValuesPerThread` (`moe_router_warp.h:96-98`) returns 0 for every
  E outside `{32,64,128,256}` and `LaunchRouterWarp` (`cuda_moe.cu:564`) returns
  `false` on `vpt == 0` before touching a tensor, so at E = 512 and E = 1024 —
  including the geometry `qwen4_exp` routes — both arms dispatch the SAME
  `MoeRouterTopKKernel<Tin,false>`. The counts stay honest, because every
  assertion really executes: 36 of the 72 rows at E > 256 are byte-identical
  repeats, and the mutation table's `12` per cell is `2 arms x 3 h x 2 dtypes`,
  of which 6 are the second count of one disagreement. COVERAGE at E > 256 is
  half what the doubled numbers imply, and only E = 256 compares two structures.

- **THE DEFERRED V-HEAD PERMUTATION'S PREFILL COST AND ITS REPAYMENT ARE
  DERIVED, NOT MEASURED (2026-09-15, commit `708a96589` and its repair).**
  `vt::VHeadPermute` is O(T * N)
  per layer per forward, and the whole cost/benefit reading above is byte
  arithmetic off the manifest: 3.26 MB moved per token over 36 layers against a
  2.649 GB per-step saving, breakeven T ~= 817, a T = 2048 prefill step
  overpaying 4.0 GB and repaying it after 1.5 decode steps. **Nothing on a
  device has confirmed any of the three numbers.** What is owed is one A/B on
  the released 67.564 GiB artifact, same build, arms interleaved: (a) the
  prefill step time at T = 2048 with the permutation and with the projections
  pre-permuted at load, (b) the decode step time on both arms, and (c) the
  end-to-end time for the reference workload (35-token prompt, 400 decodes),
  which is where the sign of the whole change is decided. Risk 4 of the issue —
  that moving these operands from cuBLAS bf16 GEMV at 162.3 GB/s to
  `QuantDotGemm*` at 93.4 GB/s may lower the achieved rate even as it lowers the
  bytes — is measured by the same run and is still the way this change could
  disappoint. Needs a CUDA build on `thor:gpu0` or `dgx:gpu0`.
- **TWO OF THE FIVE PERMUTE SITES ARE UNTESTED BECAUSE THEY ARE UNREACHED
  (same change).** `GdnBlock` (`qwen3_5.cpp:4741`) and `GdnBlockPagedMixedSpec`
  (`:5251`) each carry the out-projection-input permutation and neither is
  exercised. `GdnBlock` is called only from qwen3_5's own NON-PAGED layer loop
  (`:7804`, `:8040`), whose loaders all leave `v_head_perm_key_heads == 0`;
  `GdnBlockPagedMixedSpec` is reached from `GdnBlockPaged` (`:5414`) only under
  an active speculator at concurrency > 1, which `qwen4_exp_forward.cpp` does
  not configure. The sites exist so a future speculator or a non-paged qwen4exp
  arm cannot silently drop the permutation; a test today would gate an
  unreachable path. What is owed is the test AT THE POINT either becomes
  reachable, not before.

## Mutation record — the deferred V-head permutation repair (2026-09-15)

`thor:gpu0`, sm_110, nvcc 13.0.88, Release, Ninja, `-j 4`,
`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`.
`rc` jobs `ea9bd1ff-ede3-4d03-98be-a9afc30b1cd8` (first run) and
`9cae72c4-b2bd-4fe3-b388-28639a6e105a` (the valid one; see "The first run's
restore arm was a lie" below). Repaired tree
`aea9b99f0`, off `708a96589`.

### The one mutation, and what it convicts

M1 inverts `vt::VHeadPermute`'s TWO index maps in the CPU kernel
(`src/vt/cpu/cpu_ops.cpp`), swapping the `inverse` and forward branches. This is
the reviewer's own mutation, the one that left `test_qwen4_exp_layer_loop`
14/14 and `test_qwen4_exp_forward` 2/2 GREEN before this repair.

| suite | A2 pristine | B2 M1 applied | C2 restored |
|---|---|---|---|
| `test_gdn_v_head_permute` | 3/3 · 234/234 | **1/3 · 206/234, 2 cases RED** | 3/3 · 234/234 |
| `test_qwen4_exp_layer_loop` | 14/14 · 716/716 | **13/14 · 715/716, 1 case RED** | 14/14 · 716/716 |
| `test_qwen4_exp_forward` | 2/2 · 429/429 | 2/2 · 429/429 (see below) | 2/2 · 429/429 |
| `test_qwen4_exp_gguf_weights` | 14/14 · 3097/3097 | — | 14/14 · 3097/3097 |
| `test_qwen4_exp_qsa` | 14/14 · 7263/7263 | — | 14/14 · 7263/7263 |
| `test_qwen4_exp_gguf_load_plan` | 10/10 · 7462/7462 | — | 10/10 · 7462/7462 |
| `test_qwen3_5_gdn_spec_routing` | 13 cases, 1 RED | — | 13 cases, 1 RED |

**THE END-TO-END CONVICTION IS THE POINT.** The oracle case prints its measured
residual per arm, and the two arms separate by 155x:

```
grouped (v_head_perm_key_heads = 0): max|diff| = 0.0118021  vs a bound of 0.03
tiled   (v_head_perm_key_heads = 2): max|diff| = 0.0118021   [pristine]
tiled   (v_head_perm_key_heads = 2): max|diff| = 1.83194     [M1 applied]  RED
```

The tiled arm reproduces the grouped arm's residual against the transformers
5.16.0 golden **to every printed digit**, which is what says the deferred
permutation is the load-time one; under M1 it lands 61x ABOVE the bound. The
grouped arm stays 0.0118021 in the SAME binary under M1, so the red is the
permutation and not the fixture.

**`test_qwen4_exp_forward` STAYS GREEN UNDER M1 AND THAT IS NOT A DEFECT TO
REPAIR.** Its fixture sets no `v_head_perm_key_heads` either, and unlike the
layer-loop case it has no second arm to give one. `vt::VHeadPermute` is
therefore never called in it, so a mutation of that op cannot reach it. Reported
here rather than silently, because "green under the mutation" reads as coverage
and is the absence of it.

### The first run's restore arm was a LIE, and the mechanism is worth the line

Job `ea9bd1ff`'s arm C restored `cpu_ops.cpp` with
`tar -xzf src.tar.gz -C "$SRC" src/vt/cpu/cpu_ops.cpp`. **`tar` restores the
ARCHIVE's mtime**, which is older than the `.o` ninja had just built from the
mutant, so ninja found the target up to date, rebuilt nothing, and arm C ran arm
B's binary. `test_qwen4_exp_layer_loop` carried md5
`723eb5e3243b93ed1036940d7a3cd07f` in BOTH arms, and the source sha256 check
passed — `RESTORE_CHECK=OK` — because the SOURCE really was restored. A
source-hash restore check cannot see this; only the binary can.

Job `9cae72c4` re-ran the whole loop with `touch` before every rebuild. The
proof that the restore is now real is that **A2 and C2 are byte-identical on
every one of the seven suites**:

| suite | A2 / C2 md5 (equal) | B2 md5 (differs) |
|---|---|---|
| `test_gdn_v_head_permute` | `7f09117c5ab6f2088cd46b5ff4c58fb1` | `8c3e03a6b9c4408e7df36eada1de64c5` |
| `test_qwen4_exp_layer_loop` | `e11284f6c0c9cee23390e352356a73fa` | `4174e7c34a609106274797d77927f3ac` |
| `test_qwen4_exp_forward` | `966dd23a847addd8c19022ac44218e9b` | `83c31d72123944962feadfe57f962ac7` |
| `test_qwen4_exp_gguf_weights` | `57fcdaedcc7c2581fdc2f1fdab333631` | — |
| `test_qwen4_exp_qsa` | `86ffd3e3d62588a65026de2cc8034cb8` | — |
| `test_qwen4_exp_gguf_load_plan` | `0a184ba56a41ad040debadcdd80a1238` | — |
| `test_qwen3_5_gdn_spec_routing` | `064acac6c9241cf87c9d529b9183bc37` | — |

`SOURCE_SHA256_ORIG` and `SOURCE_SHA256_BACK` are both
`7008953ebf996ae811ca30a3a0e32e6d47347a57a75911841128c9c173146592`;
`SOURCE_SHA256_MUT` is
`5786065a9ecb435382ac681a746525708ebe62b28b500ebebfd82f54e1f91361`.

### `test_qwen3_5_gdn_spec_routing` fails on this arm, and it is NOT this change

13 cases rather than the 7 the row's earlier gate recorded, and one of them —
`GDN merged FP8 qkvz == the two split fp8 GEMMs, bitwise`,
`test_qwen3_5_gdn_spec_routing.cpp:798`, 8 assertions — is RED. The earlier
reading was a different build configuration; this is the CUDA sm_110 arm.

**It cannot be caused by this commit, and that is checkable rather than
asserted.** `git diff 708a96589..aea9b99f0 -- src include` touches three files,
and of those:

- `include/vt/ops.h` — **zero** non-comment lines changed;
- `src/vllm/model_executor/models/qwen4_exp_weights.cpp` — **zero** non-comment
  lines changed;
- `src/vt/rocm/rocm_gdn_state.hip` — relocation only: the non-comment lines of
  the two revisions are the same multiset (`grep -v '^\s*//' | sort | md5sum`
  gives `67afbbbbc1615f2fd59cf64c1af79776` on both), and a CUDA build compiles
  no `.hip` file at all.

So the object code reaching that FP8 case is identical at the base and at the
repair. The failure is pre-existing on the CUDA arm and belongs to whoever owns
the merged-FP8 qkvz path, not here.

## Mutation record — W5k (#2031)

**THE ORACLE, AND HOW IT WAS PROVED TO BE THE ORACLE.** W5j stopped rather than
guess, and it was right to; it looked in the wrong place. The lane pin is a
RELEASE, not a checkout: `transformers` **5.16.0**. W5k created a virtual
environment, installed it, and checked three things before reading a line of it:

| Check | Result |
|---|---|
| `transformers.__version__` on a live import | `5.16.0` |
| `models/qwen4_exp/modeling_qwen4_exp.py` sha256 | `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459` — the pin |
| `scripts/gen-qwen4-exp-forward-goldens.py` regenerates the committed golden | sha256 `d968a142…05d77` before and after; `git status` clean |

The third is the one that makes it an instrument rather than a download: the
environment reproduces this row's existing committed golden byte-for-byte.

**WHAT THE RUNNING MODEL SAID.** A two-step probe over the row's own fixture
geometry, at both `float32` and `bfloat16`:

| Model dtype | `conv_states[1]` (PLE ring) | `conv_states[2]` (n-gram history) |
|---|---|---|
| `torch.float32` | `float32`, `(1, 16, 9)` | `int64`, `(1, 2)`, device `cpu` |
| `torch.bfloat16` | **`bfloat16`**, `(1, 16, 9)` | `int64`, `(1, 2)`, device `cpu` |

The ring FOLLOWS the model dtype and never widens. Mechanism:
`cache_utils.py:1019-1023` allocates each slot as `torch.zeros(...,
dtype=conv_states.dtype, device=conv_states.device)` — per SLOT, from the tensor
that first reaches it — and the tensor reaching slot 1 is `hidden_states`
(`modeling_qwen4_exp.py:1157-1159`). Slot 2 is fed `input_ids.long()` (`:1070`)
at `:1089-1091`, so it is i64 on `input_ids.device`, the COMPUTE device. Both
publisher-side declarations in `MakeQwen4ExpKVCache` were correct and both
`RunQwen4ExpPleBlock` requirements were the wrong side.

The cross-step observable, over the prompt `[5,9,13,3,7,2]`, four steps:

| After | `ngram_history` |
|---|---|
| prefill | `[7, 2]` — the prompt's last two ids |
| decode 11 | `[2, 11]` |
| decode 4 | `[11, 4]` |
| decode 3 | `[4, 3]` |

A FIFO of raw token ids. INTEGERS, so unlike this fixture's bf16 activations it
cannot saturate, which is why W5k gates the cross-step path on it and asserts no
cache VALUE anywhere (W5j measured 0 of 128 indexer words and 0 of 192 paged K/V
words moving while logits moved 31.84; a rescaled fixture stays owed).

**THE RESULT.** `ModelRegistry::Forward` runs a prefill (T = 6, `past_len` 0,
sampled token 15) and then a DECODE (`past_len` 6, sampled a token) over one set
of persistent caches. Oracle golden UNMOVED at `max|diff| = 0.00982457` against
its 0.03 bound.

**THE BATTERY.** Every mutation: sha256 proved applied, build rc read BEFORE any
test output, tree restored and the restore sha256-verified against the
pre-mutation snapshot.

| # | Mutation | Build | Result |
|---|---|---|---|
| M1 | delete the n-gram history WRITE-BACK inside `RunQwen4ExpPleBlock` | rc 0 | **RED** — 3 assertions. Step-1 `CHECK( 0 == 7 )` and `CHECK( 0 == 2 )` at `:1874`; step-2 rolled-FIFO `CHECK( 0 == 15 )` at `:1909`. Also the first MEASURED reach of this block's body from a production entry point |
| M2 | seed the recurrent state WRONG on step 1 (`eos_token_id` to `0`) | rc 0 | **RED** — the oracle golden at `0.777988` against a bound of `0.03`, plus 5 assertions over 4 cases in `test_qwen4_exp_ple_block` |
| M3 | REACHABILITY: force `published` false, deleting the production route to the engine's published states | rc 0 | **RED** — step-1 history reds AND step 2 THROWS the authoritative refusal. NOT vacuous: the route is load-bearing |
| M4 | read the WRONG published slot for the history (`states[3]` to `states[1]`) | rc 0 | **RED** — refused by shape: "the recurrent group's fourth state ... must be [slots,2]" |
| M5 | the op admits bf16 but the KERNEL treats the ring as f32 — "a dtype with no kernel behind it" | rc 0 | **RED** — 65 assertions, and ONLY in the new bf16 ring case; the 10 f32 cases stayed green, so the mutation isolates the bf16 path |

**THE PREDICATE TRAP, CAUGHT IN FLOW.** The first draft refused a continuing step
on `input.multi_kv != nullptr`, while the predicate that ROUTES the PLE caches is
`g.states.size() >= 4`. Those are different: a channel can be present and still
carry a recurrent group whose `states` list was never filled — every hand-built
`GdnStateCache` in this tree does exactly that (`qwen3_5.h` documents it). The
weaker refusal would have let that case run on a zeroed per-call scratch, which
re-seeds the history every step and produces a fluent wrong answer with no error.
The authoritative refusal was moved onto the routing predicate itself; the
`multi_kv` check remains only as an early, strictly-weaker message. This is the
`refusal != route predicate` failure recorded elsewhere on this row, found before
it landed rather than after.

**WHAT W5k DID NOT PROVE.** The block's new ring-dtype EQUALITY check is a guard,
not a route: with the publisher and the stream now agreeing through one exported
constant (`kQwen4ExpStreamDType`), deleting that check reds nothing. It is
recorded as a guard rather than claimed as gated. `GPUModelRunner` was not driven
end to end — the two steps are assembled the way the runner assembles one, not BY
it. No CUDA arm was built or run; this is a CPU-only host.

## Mutation record — W5L (#2031, issue OWED)

**THE HEADLINE. A REQUEST SERVED, AND `GPUModelRunner` DROVE IT.**
`build/examples/vllm-server --model <qwen4exp gguf> --port 8733 --block-size 16
--num-blocks 64 --max-model-len 32` (no `--max-num-seqs`, so the DEFAULT applies)
answered:

```text
POST /v1/completions {"prompt":"abcdef","max_tokens":5,"temperature":0}
{"choices":[{"finish_reason":"length","text":"ppppp"}],
 "usage":{"prompt_tokens":6,"completion_tokens":5,"total_tokens":11}}
```

Six prompt tokens prefilled and five decode steps, on CPU, through
`LoadedEngine` -> `GPUModelRunner::execute_model` -> `ModelRegistry::Forward`.
Three overlapping requests at the default concurrency were all answered.

**WHAT THE RUNNER DEMANDED THAT THE HOOK LACKED: NOTHING.** This is the wave's
most useful negative result. W5j and W5k built the by-name resolution, the paged
indexer and the persistent PLE states against a hand-assembled step, and the
runner's own step needed no change to any of them. The runner's four demands were
all satisfied on the base commit, and each was VERIFIED rather than assumed:
`multi_cache_topology_` set, `multi_kv` published (5 names), all three group
tables gathered, and `consumes_multi_kv` letting the topology past
`ModelRegistry::Forward`. The only thing the ENGINE needed is in the next
paragraph, and it is a scheduling fact, not a cache one.

**WHAT DID BLOCK SERVING, AND IT IS NOT WHAT THE ROW EXPECTED.** `num_reqs > 1`
does not stop a request; it stops the SERVER. The refusal is thrown inside the
EngineCore busy loop, which treats a throw as FATAL. Measured at
`--max-num-seqs 4` BEFORE any change:

```text
engine-fatal: EngineCore busy loop threw: vt:
Qwen4ExpForConditionalGeneration: this forward serves ONE sequence per call and
the step carries 2. ... qwen4_exp_registry.cpp:254
api-server: 500 endpoint=/v1/completions ... (x3)
```

All three concurrent requests failed and the engine never served again. The
default `max_num_seqs` is 128, so that was the out-of-the-box behaviour of a
server pointed at this architecture. W5L therefore adds
`ModelFactory::serves_one_sequence_per_step`, read by
`LoadedEngine::ResolveMaxNumSeqs` AFTER the recurrent-state budget clamp (the
smaller bound must win), which prints its reduction the way the budget clamp
beside it does and returns 1. Same binary, same flags, after:

```text
INFO model concurrency: reduced max_num_seqs from 7 to 1. ...
== 1 {"text":"pppppp"}   == 2 {"text":"pppppp"}   == 3 {"text":"pppppp"}
```

The forward's refusal is UNCHANGED and still fires for a hand-built batched step
(gated by case 3 below). The batching wave itself is W5m, split and costed under
`## Owed`.

**WHAT THIS FIXTURE CAN GATE, AND WHAT IT STILL CANNOT.** Not cache CONTENT: W5j
measured 0 of 128 indexer words and 0 of 192 paged K/V words moving while the
logits moved 31.84, because this fixture's layer-3 activations sit near 2^18
where one bf16 ULP is about 1024. A rescaled fixture stays owed. The gate is the
same observable W5k found, read from a different place: the PLE n-gram history is
int64 TOKEN IDS, and W5L reads it out of the buffer the RUNNER allocated, at the
slot the RUNNER assigned (taken from group 1's own gathered table, not assumed to
be 0), after a step the ENGINE scheduled. Its step-2 read's writer is a prior
`execute_model` call. The prompt asserts its last two ids are DISTINCT, so the
roll is observable, and the case additionally asserts the history MOVED between
the two steps — without that the roll assertions pass on a prompt whose tail
happens to equal the answer.

Nothing in `tests/vllm/models/test_qwen4_exp_runner.cpp` constructs a
`ModelForwardInput`, a `CommonAttentionMetadata` or a `GDNAttentionMetadata`.

**COUNTS.** `test_qwen4_exp_runner` is new: 5 cases / 136 assertions, green.
`test_qwen4_exp_layer_loop` unchanged at 5 / 264 and the oracle golden UNMOVED at
`max|diff| = 0.00982457` against its 0.03 bound. Also rerun green on this head
because they share the fixture header or the changed engine seam:
`test_qwen4_exp_gguf_weights` 11 / 2975, `test_qwen4_exp_forward` 1 / 421,
`test_qwen4_exp_kv_cache` 5 / 414, `test_loaded_engine_dense` 30 / 128.

**THE BATTERY.** Every mutation sha256-proved applied, the build rc read BEFORE
any test output, the tree restored and the restore sha256-verified against the
pre-mutation snapshot (all three files byte-identical afterwards).

| # | Mutation | Build | Result |
|---|---|---|---|
| M1 | delete the runner's cache handoff — `forward_input.multi_kv = &multi_kv_index_` at `runner.cpp` | rc 0 | **RED** — 2 cases. The runner-driven step and the `LoadedEngine` generate both throw "the runner handed 2 paged K/V caches for 1 qwen_sparse_attention layers, and no by-name cache index". The handoff is what carries the whole three-group topology |
| M2 | make the step read the WRONG sequence's state — `caches.ple[i].state_row = row + 1` | rc 0 | **RED** — 4 assertions. The history at the runner's assigned slot stays `{0,0}` while the prompt's tail is `{7,2}`; the FIFO roll assertion reds at `0 == 15`; `h1 != h2` reds at `{0,0} != {0,0}`. In the engine case the block's own bound check throws, because a clamped engine has ONE slot |
| M3 | REACHABILITY: delete `gather_group_block_tables(num_reqs)`'s production call site | rc 0 | **RED** — 2 cases, "the engine gathered no block table for published group 2 ... 0 of 3 published group(s) carry one". NOT VACUOUS: the gather is load-bearing for both the runner-driven step and the engine-driven generate |
| M4 | REACHABILITY: delete the clamp's production call site — pass `false` instead of `factory->serves_one_sequence_per_step` in the `LoadedEngine` ctor | rc 0 | **RED** — exactly 1 assertion, `eng->max_num_seqs()` reads 3 not 1. The generate still SUCCEEDS, which is the point: with one request in flight the defect is invisible, and only the resolved-concurrency assertion sees it |

**WHAT W5L DID NOT PROVE.** No CUDA arm was built or run — there is no CUDA
kernel for any `qwen4_exp` op, so a device run is unavailable rather than
unmeasured. No published checkpoint was served; every byte came from the
synthetic fixture, so there is no token number and no speed number.
`/v1/chat/completions` was NOT served: the fixture's 16-token vocabulary is
`'a'..'p'` and the fallback role-join prompt contains `user`, so it returns
`500 tokenizer: symbol "u" not in vocab`. That is the fixture's limit and it is
recorded under `## Owed` rather than worked around, because widening the vocab
moves every value the oracle golden measures. And `num_reqs > 1` is still refused:
what changed is that the refusal no longer kills the engine.


## Mutation record — W5p (#2031, issue OWED)

The wave that makes a **quantized** hyper-connection mix weight run, so the
released `unsloth/Qwen3.8-Flash-Next-GGUF` can prefill. Base `c45ecce47`
(`row/MODEL-MM-QWEN4-EXP-W5N`), branch `row/MODEL-MM-QWEN4-EXP-W5P`, CPU host,
`cmake -G Ninja` with no `CMAKE_BUILD_TYPE` (so `NDEBUG` is NOT set and asserts
are live), `-j 2`.

### The RED, verbatim

`test_qwen4_exp_hc_device` at the pre-wave head, with the new Q8_0 case added and
nothing else changed. Build rc 0, zero warnings, read BEFORE the test output:

```text
tests/vllm/models/test_qwen4_exp_hc_device.cpp:831: ERROR: test case THREW
exception: vt: qwen4_exp_gated_residual: input_mix_weight_down must be float
(f32/bf16 for outputs) at src/vt/ops.cpp:2552
[doctest] test cases:  11 |  10 passed | 1 failed | 0 skipped
```

That is the SAME string, from the same line, that W5n recorded from the released
checkpoint on `thor:gpu0`.

### The measurements the gate carries

| Quantity | Value |
|---|---|
| f32 arm vs the in-test double reference, `mixed` | `1.00553e-07` |
| f32 arm vs the in-test double reference, `injection` | `4.4584e-08`, both under `kTol` 1e-5 |
| Q8_0 arm vs the same double reference, `mixed` | `0.00249794` |
| Q8_0 arm vs the same double reference, `injection` | `0.00169157`, against a stated bound of `5e-3` |
| Q8_0 arm vs f32 arm, same logical weights | `0.00249791`, asserted **> 0** |
| transformers 5.16.0 end-to-end golden | `max\|diff\| = 0.00982457` against 0.03 — **UNMOVED** |

The Q8_0 residual is the ACTIVATION encoding and not weight error, because the
weights are chosen `d * q` for an f16-exact power-of-two scale and an int8 code,
so `dequant(quant(w)) == w` to the bit. The `> 0` assertion is the one a
"dequantize the mix weights at load" workaround fails: with a lossless weight
encoding the two arms would be BIT-IDENTICAL, and only the quantized route
introduces an activation encoding to separate them.

### Mutations

Every row: mutation applied and proved by a `sha256sum` that differs from the
recorded baseline, build return code read BEFORE any test output, tree restored
with `git checkout --` and proved byte-for-byte by `sha256sum -c` against the
baseline file plus an empty `git diff HEAD`.

Baseline: `src/vt/ops.cpp` `42719dfc…`, `src/vt/cpu/cpu_qwen4_exp.cpp`
`622d9fd5…`, `tests/vllm/models/test_qwen4_exp_hc_device.cpp` `4415614f…`.

| # | Mutation | Applied sha256 | Build | Result |
|---|---|---|---|---|
| M1 | **REACHABILITY.** `check_projection(mix_down/mix_up, …)` restored to `check_operand(…, false)` — the pre-wave contract, nothing else touched | ops.cpp `d0a01aad…` | rc 0 | **RED in TWO suites.** `test_qwen4_exp_hc_device` 10/11, the Q8_0 case throwing the verbatim refusal above. `test_qwen4_exp_layer_loop` 5/6, and the failure is `REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in))` throwing that SAME string — so a block-typed mix weight really does reach this op through the production entry point on a loaded GGUF, and the end-to-end case is not vacuous. Every other case in both suites, the transformers golden included, stayed green |
| M2 | **THE ROUTE PREDICATE, one way.** `if (false && IsBlockQuant(w.dtype))` in `ProjectRow` — a quantized weight forced down the float pointer walk | cpu_qwen4_exp.cpp `570ddd04…` | rc 0 | **RED in both suites**, `vt: qwen4_exp_gated_residual: unsupported input dtype at src/vt/cpu/cpu_qwen4_exp.cpp:87` — the `LoadF32At` default, which is the scalar element walk the fusion forced |
| M3 | **THE ROUTE PREDICATE, the other way.** `if (true \|\| IsBlockQuant(w.dtype))` — a FLOAT weight forced down the quantized path | cpu_qwen4_exp.cpp `69864e4f…` | rc 0 | **RED on 7 of 11 cases**, every golden among them: `vt: matmul_bt_quant: weight must be a block-quantized dtype (use MatmulBT for elementwise weights)`. The predicate is load-bearing in both directions |
| M4 | **ONE BLOCK'S SCALE CORRUPTED.** In the fixture builder, block 0's stored f16 scale is written as `2d` while the logical f32 weight keeps `d`, so 32 weight elements decode at twice their value. The builder's own exactness `REQUIRE` is lifted in the same mutation so the corruption reaches the op instead of aborting the case | test file `82851d6e…` | rc 0 | **RED on 3 assertions**, `max\|mixed − double ref\| = 0.204279` and `max\|inj − double ref\| = 0.36377` against the 5e-3 bound, and the arm-vs-arm check at `0.204279`. An **82x** separation from the honest `0.00249794`, so the bound discriminates a single wrong block scale rather than merely admitting the encoding error |
| M5 | **THE ELEMENTWISE HALF OF THE POLICY.** `hc_norm_w` routed through `check_projection` — the gamma loosened to accept blocks | ops.cpp `3570b5ac…` | rc 0 | **RED**, and it is the two-sided form that matters: the gamma gets PAST the named refusal and dies deeper, `threw a DIFFERENT exception! (contents: "…unsupported input dtype at cpu_qwen4_exp.cpp:87")`. A bare `CHECK_THROWS` would have stayed green here, which is why the case asserts on the message |

After restore, rebuild rc 0 and all three suites green: `test_qwen4_exp_hc_device`
11/11 (516 assertions), `test_qwen4_exp_gguf_weights` 12/12 (3074),
`test_qwen4_exp_layer_loop` 6/6 (309) with the golden at `0.00982457`.

### What is NOT proved

Nothing had run the RELEASED checkpoint through the repaired path AT W5p. W5n's
run needed `thor:gpu0` and 4446 s to load; W5p is a CPU wave with no lease and
did not attempt it. The claim here is that the op, the loader and
`ModelRegistry::Forward` all carry a Q8_0 mix weight on the miniature, and that
the refusal the released file hit is gone at its source.
### W6-CUDA — the first CUDA arms this architecture has ever had

**THE SPLIT, AND THE CRITERION IT WAS MADE ON.** Six `qwen4_exp` ops were
CPU-only, plus `vt::RmsNormGroup` and the block-decoding n-gram gather. This wave
gives CUDA arms to **three**: `vt::Qwen4ExpPleConv`, `vt::Qwen4ExpPleGate` and
`vt::Qwen4ExpGatedResidualWriteBack`.

The line is not leverage and it is not convenience. It is **whether the op
performs a reduction across a parallel axis**, because that is precisely the
question every `## Owed` CUDA entry above already poses:

| op | reduction across a parallel axis | decision a device arm must make | this wave |
|---|---|---|---|
| `vt::Qwen4ExpGatedResidualWriteBack` | none — one multiply, one add per output | none | **done** |
| `vt::Qwen4ExpPleGate` | none — elementwise | none | **done** |
| `vt::Qwen4ExpPleConv` | four taps, walked by ONE thread in the host's order | none | **done** |
| `vt::Qwen4ExpGatedResidual` | grouped sum of squares, `double` here | the reduction WIDTH: a 571x separation from f32 at group size 2560 | owed |
| `vt::RmsNormGroup` | grouped sum of squares, **f32** here, in the dumped order | same question, opposite answer — the two must NOT be unified | owed |
| `vt::Qwen4ExpQsaCompress` | pooled-key sum of squares, f32 ascending | the width, plus the `round_intermediates_to_bf16` arm | owed |
| `vt::Qwen4ExpQsaGatherAttention` | two softmax passes over a gathered prefix | the VISIT ORDER (ascending is what makes a sub-budget gather bit-identical to dense), a DEVICE-side `keys_visited`, and gather-vs-mask | owed |

The three done ops inherit their CPU arms' recorded precision contracts
**unchanged** — the conv's `double` four-tap accumulator, the gate's all-double
interior and its `SignedSqrt` NaN guard — so no wave has to make a decision on
their behalf and none was made. The four owed ops each own a decision this spec
already records and this wave did not pre-empt.

**NOTHING IN PRODUCTION REACHES THESE THREE KERNELS, AND THAT IS NOT A SPLIT
ARTEFACT.** `ModelRegistry::Forward` is all-or-nothing: a `qwen4_exp` step calls
all six ops plus `vt::RmsNormGroup` plus a block-decoding `vt::Embedding` gather,
and `GetOp` THROWS on an unregistered (op, device) rather than falling back —
the portable CPU reference tier cannot rescue it, because that tier is gated on
`Backend::DeviceMemoryIsHostAddressable()` and `CudaBackend` leaves it at the
base `false` (CUDA on GB10 allocates with `cudaMalloc`; #844, #1435). So **no
split short of all six plus `vt::RmsNormGroup` plus `EmbeddingKernelCuda`'s
missing keep-quant arm makes `--device cuda` run this model**, and a wave that
had written all seven blind on a host with no CUDA compiler would have been
guessing at four recorded decisions at once. The reachability of these three arms
from a production entry point is therefore **VACUOUS, not proven**, under
AGENTS.md "Nothing lands dead": the wiring is owned by row
`MODEL-MM-QWEN4-EXP` under campaign
[#1978](https://github.com/mudler/vllm.cpp/issues/1978), tracked by
[#2031](https://github.com/mudler/vllm.cpp/issues/2031), and its own issue is
OWED (GitHub writes are `403` from this host, account suspended, so nothing could
be filed and no row was appended to `.agents/issue-index.md`; an index row
pointing at an issue that does not exist is worse than an absent one).

#### Evidence, and the exact boundary of what it covers

**THE KERNELS COMPILE AND RUN ON A GPU. This paragraph replaces one that said
they never had**, and the replacement is the point: an out-of-date warning is
its own defect. The device is `thor:gpu0`, a Jetson Thor at **`sm_110`**, on
2026-08-30 23:01-23:35 UTC. Toolkit `nvcc` 13.0.88, configure
`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110`. Both translation units
built -- `[3/630] cuda_qwen4_exp.cu.o`, `[4/630] cuda_qwen4_exp_ple.cu.o` -- with
**zero** lines matching `error:|Error [0-9]|FAILED` in `build-w6.log`, which ends
at `[630/630] Linking`. Logs: `/workspace/q4exp-w6cuda/out-thor/`.

**THE BUILD RC IS DERIVED, NOT READ, and that distinction is kept.** The literal
`### W6 BUILD RC=` line goes to the job's stdout, and that job has since aged out
of `rc jobs`, so it cannot be quoted. What can be shown is control flow:
`run-thor.sh:90-93` prints the rc and then `if [ "$bld" -ne 0 ]; then ... exit
94`. Everything the script writes after that point exists -- `gate.log`, six
`mut-*` logs, `build-final.log`, `gate-final.log` -- so the branch was not taken
and `bld` was 0. Two further corroborations: ninja prints a terminal `[630/630]`
only when every step succeeded, and `tests/test_qwen4_exp_cuda` was subsequently
EXECUTED, which is impossible unless it linked. This is a stronger argument than
"no error lines", which is an absence-of-evidence claim, but it is still a
derivation and is labelled one.

**`sm_121a` (GB10) IS ALSO COVERED, AND IT IS THE CLEANER OF THE TWO RUNS.**
`dgx:gpu0`, nvcc 13.0.88, `-DVLLM_CPP_CUDA_ARCHITECTURES=121a`, 2026-08-31. Here
the build rc is READ rather than derived, which closes the one soft spot in the
`sm_110` evidence:

```
### CONFIGURE RC=0        ### BASELINE BUILD RC=0    ### W6 BUILD RC=0
### GATE RC=0             ### FINAL BUILD RC=0       ### FINAL GATE RC=0
```

**12 cases, 12 passed. 351 assertions, 351 passed. `Status: SUCCESS!`** -- and
the same after the mutation battery restored the tree, which is what
`### FINAL GATE RC=0` says. The count is 351 rather than `sm_110`'s 323 because
this run carried the re-derived oracle bound and its bitwise backstops; the one
assertion that failed on `sm_110` was that bound, and with it corrected the suite
is green on both arches.

`cuobjdump` confirms the objects are genuinely built for this architecture, which
no rc can show on its own: `cuda_qwen4_exp_ple.cu.1.sm_121a.cubin`.

**The gated tree is pinned to a commit, not merely described.** The job printed a
sha256 for each file it applied, and all eleven match this branch's `e9862d864`
byte for byte (`sha256sum -c`, 11/11 OK). So "the gate passed" and "the gate
passed on the code in this commit" are the same statement here.

The kernels being arch-invariant by inspection -- zero occurrences of `mma.sync`,
`wmma`, `__CUDA_ARCH__`, inline `asm`, CUTLASS, `ldmatrix`, `cp.async` or any
`sm_*` literal, and only IEEE round-to-nearest intrinsics plus sm_80+ converters
-- is now corroborated rather than merely argued: two architectures, three
generations apart in the feature table, produce output that is bitwise identical
to the same CPU arms.

### Mutations on `sm_121a`, with applied-proof and restore-proof

Every mutation printed a `sha256 before=... after=...` pair proving it changed
the file, and a `RESTORED byte-for-byte` line proving the tree came back. Five
red, and the sixth is a compiler proof:

| mutation | build rc | run rc | reading |
|---|---|---|---|
| M1 dilation ignored | 0 | **1** | RED |
| M2 NaN guard dropped | 0 | **1** | RED |
| M3 write-back contracts into an fma | 0 | **1** | RED — the byte-identity claim is load-bearing on GB10 too |
| M4 conv accumulates in float | 0 | **1** | RED |
| M5 kCUDA registration deleted | **1** | — | BUILD FAILED: a COMPILER proof, not a test verdict. `-Werror` refuses the orphaned kernel, so no test ran. Not counted as a red |
| M6 ring write-back dropped | 0 | **1** | RED |

**The `sm_110` mutation counts and the `sm_121a` ones agree**, including M5
failing to build on both. That is two independent devices reporting the same
battery.

### The other suites on `sm_121a`

`### BASELINE qwen4 ctest RC=8` before the change and `### qwen4 ctest RC=8`
after, with an IDENTICAL failing set both times -- `test_qwen4_exp_gguf_load_plan`,
`..._gguf_weights`, `..._layer_loop`, `..._runner`, `..._forward` -- and
`test_qwen4_exp_cuda` **absent from it**, because it passed. The change regressed
nothing and its own suite is green.

### What the sm_110 run measured

`gate.log`, 12 cases, 323 assertions, **322 passing**. Against the CPU arms the
device output is BITWISE identical everywhere it is compared:

| device gate, `sm_110` | result |
|---|---|
| conv vs the transformers ORACLE, dilations 1 / 2 / 3 | 5.96e-08 / 5.96e-08 / 2.98e-08 |
| conv vs the CPU arm, output, 4 token counts, non-zero ring | **0 differing — bitwise** |
| conv vs the CPU arm, RING write-back, all four | **0 of 144 — bitwise** |
| conv under catastrophic cancellation | **0 of 64 — bitwise**; the `double` accumulator holds on device |
| gate vs the CPU arm at T=17, hc=4, hidden=129 | **0 of 8772 — bitwise** |
| write-back vs the CPU arm, 1x2x1 / 3x4x8 / 7x3x129 / **2x4x2560** | **0 of 2 / 96 / 2709 / 20480 — bitwise** |
| all 18 write-back dtype triples | **0 of 204 each — bitwise** |
| all 6 conv dtype pairs | **0 of 224 each — bitwise** |
| all 6 gate dtype pairs | **0 of 693 each — bitwise** |

The 2x4x2560 row is the released hyper-connection geometry, and it is where the
`__fmul_rn`/`__fadd_rn` byte-identity claim either lands or does not. It lands.

**The one failing assertion was this suite's own bound, and the fix is a
re-derivation rather than a widening.** The gate's oracle case missed at
4.76837e-07 against 4.37555e-07. The CPU arm misses the SAME golden by the SAME
4.76837e-07 on the same 36 of 96 elements, while the two arms are bitwise equal
to each other on all 96 -- so a bound the CPU arm also fails is a statement about
the bound. `kUlpTol` is an ARM-VS-ARM constant; a torch-dumped golden is an
independent f32 computation that neither arm is within one ulp of, and
`test_qwen4_exp_ple_gate.cpp:94` has always used 1e-5 for this comparison. Every
oracle case is now backstopped by a bitwise CPU-vs-CUDA comparison on the same
input, and the pairing is measured: under the float-accumulator mutation the
three conv oracle cases go green at the new bound (1.19e-07, 5.96e-08, 2.98e-08)
and the three backstops red bitwise on 90, 71 and 86 of 192. The backstop
recovers exactly what the bound gives up, as an equality rather than a tolerance.

### Mutations, on the device

| mutation | build | result on `sm_110` |
|---|---|---|
| M1 the conv IGNORES `args.dilation` | rc 0 | **RED**, 30 assertions; oracle dilations 2 and 3 wrong by 0.982 and 0.563 |
| M2 the `SignedSqrt` NaN guard DELETED | rc 0 | **RED**, 2 assertions beyond the pre-existing one — the NaN case, which no tolerance can reach |
| M3 the write-back contracts into an fma | rc 0 | **RED**, 8 byte-gate assertions. The byte-identity claim is load-bearing on a real device, not only in theory |
| M4 the conv accumulates in FLOAT | rc 0 | **RED**, 21 assertions; the designed fixture reads 0.731059 against a 8.77e-08 bound |
| M5 the `kQwen4ExpPleConv` kCUDA registration DELETED | **BUILD FAILED** | a COMPILER proof, not a test verdict: `1 error detected in the compilation of cuda_qwen4_exp_ple.cu`. Deleting the registration orphans the kernel and `-Werror` refuses it. Recorded as withdrawn-and-informative rather than counted as a red |
| M6 the conv's RING write-back dropped | rc 0 | **RED**, 11 assertions, all of them ring gates — the output gates stayed green, which is the separation the ring is gated apart FOR |

Every failing log also carries the pre-existing `4.76837e-07 <= 4.37555e-07`
line, which is the bound defect above and not a mutation effect; the counts here
have it subtracted.

### The other suites, and what is NOT this wave's

`ctest -R qwen4_exp` on the **baseline** tree, before the change was applied,
already failed five: `test_qwen4_exp_gguf_load_plan`, `..._gguf_weights`,
`..._layer_loop`, `..._runner`, `..._forward`. After the change the failing set
is **those same five plus `test_qwen4_exp_cuda`**, this wave's own suite with its
one bound assertion. **The change broke nothing.** `ctest -R cuda` additionally
reports `test_cuda_ops` and `test_ops_matmul_fp8_block_cuda`, both recorded as
pre-existing reds on this device in `.agents/environment.md` (#1802 and #1725
respectively). Neither is reachable from anything this wave touched.



**THE LEASE ATTEMPT, because "no device" should say what was tried.** A
`dgx:gpu0` job was submitted at the start of the wave and sat at queue position
**#1 for roughly three hours**, behind the developer's own `dflash2-staged`
runs, which finished and re-queued more than once in that window. A `thor:gpu0`
job was added later as the developer's named fallback, reached position #3
behind the sibling W5n released-checkpoint run and two of the developer's
`thor-parity` jobs, and was CANCELLED rather than left armed — a queued job
nobody is watching fires whenever the device frees and takes a box another wave
is waiting for. It was killed while still queued and never started, so it cost
the fleet nothing.

The dgx job did not run either. It was cancelled deliberately, with
`RC_SUBMITTER=w6cuda@qwen4exp rc kill`, and the same script re-submitted under
the default identity so that the job would be killable by a plain `rc kill`. The
rule that motivated that swap is recorded once, below, rather than argued here.

**What WAS measured, and why it is worth having.** The two `.cu` files were
compiled and EXECUTED on the host under a shim that makes `__global__` a plain
function and the launch indices a single-thread grid, so every grid-stride loop
walks its whole index space serially. `RegisterOp` was stubbed to CAPTURE what
each Registrar registers, which is how the driver reaches kernels that live in
anonymous namespaces, and BOTH arms were then driven through the same function
pointers the dispatcher would hand a caller. This exercises the arithmetic and
the INDEXING — a transposed stride, a wrong tap lag, a ring read-after-write
hazard, a swallowed NaN — and it exercises nothing CUDA-specific: not a launch,
not memory, not a generated instruction, not `__fmul_rn` versus a contracted fma.

| host simulation, tree `ad436f49` | result |
|---|---|
| conv vs the transformers ORACLE, dilation 1 / 2 / 3 | `max abs diff` 5.96e-08 / 5.96e-08 / 2.98e-08 |
| conv vs the CPU arm, output, tokens 1 / 4 / 9 / 12, NON-ZERO incoming ring | **0 of 16 / 64 / 144 / 192 elements differ — bitwise** |
| conv vs the CPU arm, RING write-back, same four | **0 of 144 differ — bitwise**, all four |
| conv double-accumulator fixture (taps 1.0, 2^40, -2^40, 0) | cpu 0.731058598, sim 0.731058598, double answer 0.731058579; an f32 accumulator gives **0** |
| gate vs the transformers ORACLE | `max abs diff` 4.77e-07 |
| gate vs the CPU arm at T=17, hc=4, hidden=129 | **0 of 8772 differ — bitwise** |
| gate NaN arm | `out[0]` is NaN, not the `0.5 * value = 1.0` a dropped guard returns |
| write-back vs the CPU arm, 1x2x1 / 3x4x8 / 7x3x129 / **2x4x2560** | **0 of 2 / 96 / 2709 / 20480 differ — bitwise**, at the released hyper-connection geometry |
| write-back hc/hidden stride, structurally, with hc == hidden | 0 of 48 misplaced |

**DTYPE COVERAGE, AND A SECOND INSTRUMENT DEFECT.** The f32 table above leaves
the runtime dtype TAG's bf16 and f16 arms completely untouched, and a wrong tag
mapping there would be invisible to every case in it. Thirty-two further
combinations were therefore run: all **18** write-back `(hyper, block, injection)`
triples the op admits (`hyper` is an output, so f32/bf16 there), all **6** conv
`(x/weight, state/out)` pairs, and all **6** gate `(value, out)` pairs. **Every one
is bitwise equal to the CPU arm — 0 of 204, 0 of 224 and 0 of 693 elements
differing respectively.**

Getting there cost a second instrument defect worth recording beside M3's. The
first run of that table failed all 27 non-f32 cases with `max|diff|` up to
**2.4e9**, which reads exactly like a catastrophic kernel defect. It was not: the
lint shim's `__bfloat162float` and `__half2float` were stubs returning the raw
16-bit pattern cast to float, so a bf16 `0x3F80` arrived as 16256. Pointing them
at the tree's own `vt::BF16ToF32` / `vt::F32ToBF16` / `vt::F16ToF32` /
`vt::F32ToF16` turned all 27 green with no change to any kernel. Broken
instruments fail toward a code verdict, and this one nearly convicted three
correct kernels.

**That substitution is also the sharpest limit on this whole simulation** and it
is stated rather than buried: using the host helpers in place of the CUDA
intrinsics means the simulation ASSUMES `__float2bfloat16 == vt::F32ToBF16` and
`__float2half == vt::F32ToF16` instead of testing it. Both are documented
round-to-nearest-even and `cuda_ops.cu` already asserts the equality in a comment,
but only a device run confirms it — and the whole bf16 store path of these three
kernels rests on it.

**Mutations, on the host simulation.** Each was applied with a sha256 before/after
pair proving it changed the file, built with the rc read FIRST, run, and the tree
restored (`git status` clean at `ad436f49`). All six build, so none is withdrawn.

| mutation | build | result |
|---|---|---|
| M1 the conv IGNORES `args.dilation` | rc 0 | **RED.** Oracle dilation 2 and 3 both wrong (`max abs diff` 0.982 / 0.563), CPU comparison wrong at every token count (up to 6.74). Dilation 1 correctly stays green, which is the control: the mutation hard-codes 1 |
| M2 the `SignedSqrt` NaN guard is DELETED | rc 0 | **RED, and it reproduces the recorded hazard exactly**: `out[0]` reads **1.0**, which is `0.5 * value` — a poison value rendered as a plausible number. No tolerance can see this; only the NaN case can |
| M3 the write-back contracts into an fma | rc 0 | **RED — but only after an instrument defect was found.** See below |
| M4 the conv accumulates in FLOAT | rc 0 | **RED** on the designed fixture (sim reads **0** against the double answer 0.731) and on all four random cases (7/16, 31/64, 75/144, 96/192 elements differ) |
| M5 the `kQwen4ExpPleConv` kCUDA registration is DELETED | rc 0 | **RED**, abort at the lookup: the op is simply absent from the table. In the built gate this is `GetOp` throwing, which is what the registration case asserts |
| M6 the conv's RING write-back is dropped | rc 0 | **RED ON THE RING GATE ONLY.** Every output comparison stayed GREEN and only the four ring comparisons fired. This is the separation the ring is gated apart from the output FOR: a kernel that computes every output correctly and leaves the cache unshifted is wrong on the NEXT step and a value-only gate cannot see it |

**M3 IS THE FINDING WORTH READING TWICE, AND IT IS AN INSTRUMENT DEFECT, NOT A
KERNEL ONE.** At `-O1` the mutated write-back compiled and the simulation stayed
GREEN — 0 of 20480 elements differing at the model geometry — which reads exactly
like a surviving mutation. It was not: `objdump` found **zero** `vfmadd`
instructions in that object. gcc had not contracted anything, so the mutation was
inert and "survived" meant "never took effect". At `-O3 -mfma` the same object
carries **2** `vfmadd` instructions and the mutation REDS: 28 of 96, 634 of 2709
and **4459 of 20480** elements differ, `max|diff|` 4.77e-07. The applied-proof for
this mutation is therefore the FMA COUNT (0 -> 2), not the sha256, because the
sha256 was already correct while the mutation did nothing. The unmutated control
was re-run at the same `-O3 -mfma` and reads 0 mismatches, so the flag change is
not what turned it red. This is the [[mutation-build-failure-reads-as-a-passing-test]]
family in a third guise — the build succeeded, the mutation applied, and the
COMPILER declined to express it.

It also has to be said that the M3 emulation needed the two sides compiled with
DIFFERENT flags — the mutated TU with `-ffp-contract=fast -mfma`, the CPU arm with
the project's pinned `-ffp-contract=off` — because compiling both with contraction
makes them agree again and hides the very asymmetry the real build has (host
pinned off, nvcc `-fmad` on and unpinned). A single-flag emulation is not a test
of this property.

**TWO `rc` FACTS THIS WAVE PAID FOR, BOTH ABOUT WHO OWNS A JOB.** Neither is in
`.agents/environment.md` and both cost this wave its queue position.

1. **`rc run --as <name>` makes the job unkillable by you.** The submitter it
   records is the `--as` value, and a later plain `rc kill` answers
   `not_job_owner: only the submitter or an admin may kill this job`. The escape
   is `RC_SUBMITTER=<same name> rc kill <id>`, which works — but a job you cannot
   cancel from the shell that made it is one that fires unattended on a shared
   box. Submit plainly.
2. **`rc run` CANCELS ITS OWN QUEUED JOB WHEN THE CLIENT DIES.** It is not
   fire-and-forget. This wave's job reached queue position #1 on `dgx:gpu0` and
   was then cancelled outright — `rc: cancelled queued job 7d58cbb7...`, and
   `rc jobs` records `killed (killed by mudler@mudler-ubuntu-box)` — because the
   streaming client was stopped. Nothing about the state of the DEVICE changed;
   the client's death was the whole cause. A submission that has to outlive the
   shell that made it therefore needs the client detached, and the results read
   back from `/workspace` or `rc logs` rather than from the stream:

   ```sh
   setsid nohup rc run -d dgx:gpu0 --max-runtime 3h -- bash /workspace/<dir>/run.sh \
       > run.log 2>&1 < /dev/null &
   ```

   **Verify that it took, because nothing in `rc ps` shows this hazard.** The
   client must report `ppid=1` and a session id equal to its own pid
   (`ps -o pid,ppid,sid -p <pid>`). Anything else still shares a session with the
   shell that launched it and is one reap away from cancelling its own job.

The practical cost was two full queue traversals on a box whose queue ran four to
six deep, so this is recorded as an environment fact rather than as an anecdote.

**Still owed after this wave, in order:**

- **The CUDA arms of the four reduction ops**, each with the decision named in the
  table above. Unchanged by this wave except that the precedent for HOW a
  `qwen4_exp` device arm is written, gated and mutated now exists.
- **`EmbeddingKernelCuda` decodes no blocks.** `src/vt/cuda/cuda_ops.cu` refuses a
  block-quantized table BY NAME (f32/bf16 only) while the CPU `EmbeddingKernel`
  carries the keep-quant arm that holds the 51.2 G-parameter n-gram table at
  28.8 GB of IQ4_NL instead of 102.4 GB of bf16. The n-gram gather therefore
  cannot run on CUDA even before the six ops. This entry already existed above;
  it is restated here because it is now one of the LAST things between this
  architecture and a device step, rather than one of many.
- **The MoE adapter is still rebuilt per layer per step**, which on a device arm
  loses `ResidentWeight::d_dev` and re-uploads the tower. Hoisting it to load time
  remains owed WITH the remaining CUDA arms, and no speed claim on this row is
  admissible before it. **This wave makes NO speed claim and measured none.**
- **f16 is admitted by these three device arms and has no ORACLE, only a CPU
  comparison.** The op contract admits f32/f16/bf16 and the runtime-tag design
  made admitting f16 free, so it is admitted rather than refused — a device arm
  that refused a dtype its CPU sibling accepts would be a divergence to record.
  The CPU-vs-CUDA half is now GATED: `test_qwen4_exp_cuda.cpp` walks all 18
  write-back `(hyper, block, injection)` triples, all 6 conv `(x/weight,
  ring/out)` pairs and all 6 gate `(value, out)` pairs, each held to BITWISE
  equality, because widening on load and rounding once on the store is the same
  operation on both arms and anything else is a defect rather than a dtype cost.
  What is still owed is the ORACLE half: the transformers goldens are f32 and the
  model dtype is bf16, so nothing upstream has ever been run at f16 for these
  ops. Owed: an f16 golden, or a recorded statement that no caller produces one.
- **The dtype tag is a runtime switch, not a template parameter**, which is a
  deliberate divergence from the `<Tin, Tout>` house style of `cuda_ops.cu`
  argued in each TU's header (the tag is a kernel-wide scalar, so the branch is
  warp-UNIFORM). Templated specialisations are a SPEED item and are owed with the
  MoE hoist above; nothing here has been measured for throughput.
- **The QSA indexer's page translation is still a HOST read.** `vt::Qwen4ExpPleConv`'s
  device arm discharges the SMALLER instance of that problem — it reads
  `query_start_loc` and `conv_state_indices` on the device rather than on the
  host — so the pattern the QSA arm needs now has an in-tree precedent on this
  row. The QSA entry itself is unchanged and still owed.


**W5q ATTEMPTED IT AND THAT LAST CLAIM HELD.** On the composed W5p + LOAD-IO
tree the released artifact prefills and decodes without throwing, and
`POST /v1/completions` returns 200 rather than W5n's 500. What W5q found instead
is a degenerate forward — eight id-0 tokens, byte-identical across two prompts.
That did not weaken W5p's claim, and W5p's claim never covered it: the op-level
and miniature gates say nothing about the values the released geometry produces.

**W5s NAMED THE CAUSE AND W8CONFIRM ISOLATED IT.** The composed tree W5q
ran predates W5r, so on `thor` — aarch64 i8mm, where
`vt::cpu::QuantRepackActive()` is TRUE — the shared `dense_attn::ResidentWeight`
was still dropping the repack marker and `kMatmulBTQuant` was reading
`block_q8_0x4` buffers as flat `q8_0` on every hyper-connection mix weight. That
produced a NaN in layer 0 which propagated to an all-zero logit row, and `argmax`
over a row with no maximum returns index 0. On `origin/main` `52f7ccbfc`, which
carries W5r, the same artifact on the same box answers `" Paris. Given this
fact, what is"` and `" 100°C at sea level"`. **W5s ASSERTED W5r WAS THE CAUSE ON
EVIDENCE THAT COULD NOT SEPARATE IT FROM W5p**, because its comparison spans
`701606e51` to `52f7ccbfc` and both commits land inside it. W8CONFIRM closes
that by building two binaries from ONE `52f7ccbfc` tarball differing only in
W5r's two lines: with them deleted the SAME tree returns
`"!!!!!!!!!!!!!!!!"` again, and disabling the repack chain on that same
defective binary restores coherent output. See the W5s and W8CONFIRM entries
under `## Owed`.

### W6-CUDA-B — the four reduction arms W6-CUDA left owed

**WHAT THIS WAVE LANDS.** CUDA arms for the FOUR `qwen4_exp`-path ops that had
none: `vt::Qwen4ExpGatedResidual` (the mixer), `vt::RmsNormGroup`,
`vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention`. The mixer joins
its write-back sibling in `src/vt/cuda/cuda_qwen4_exp.cu`; the two QSA arms land
in a new `src/vt/cuda/cuda_qwen4_exp_qsa.cu` mirroring
`src/vt/cpu/cpu_qwen4_exp_qsa.cpp` one for one; the norm lands in a new
`src/vt/cuda/cuda_rms_norm_group.cu` rather than joining `cuda_layernorm.cu`,
which is a surface several rows write and which AGENTS.md "Records" therefore
calls a lock. Gates: `tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp` and
`tests/vt/test_ops_rms_norm_group_cuda.cpp`.

This is the direct continuation of W6-CUDA, whose split table above names each of
these four under "the decision a device arm must make". This wave MAKES those
decisions and records them below rather than inheriting them.

**IT WAS DISPATCHED AS THREE OPS, AND THE COUNT WAS WRONG IN A WAY WORTH
RECORDING.** The dispatch brief said four of the six `qwen4_exp` ops already had
CUDA arms and three did not. The audit that produced it counted SUBSTRING matches
over the registration sites, and `kQwen4ExpGatedResidualWriteBack` CONTAINS
`kQwen4ExpGatedResidual` — so the write-back's registration made the mixer read
as covered. It was not. `grep -rn 'RegisterOp(OpId::kQwen4Exp' src/vt/cuda/` at
`9fb40279d` returns exactly three lines, and
`src/vllm/model_executor/models/qwen4_exp_forward.cpp:418`, `:476` and `:535`
call the mixer on the production path. Two independent audits reached the same
conclusion before any code was written. **The general shape is the one this file
keeps meeting: a check whose predicate is weaker than the property it is asked
about returns a confident wrong answer.** An op-name audit must anchor the match,
because op ids in this tree are deliberately built as extensions of one another.

**AND IT STILL DOES NOT UNBLOCK `--device cuda`, WHICH IS THE SECOND THING TO
SAY.** `ModelRegistry::Forward` is all-or-nothing and `EmbeddingKernelCuda` still
refuses a block-quantized table by name, so the block-decoding n-gram gather has
no CUDA arm and no `qwen4_exp` step can reach a CUDA queue. That gather is a
separate wave's file territory and is deliberately untouched here. Separately,
`IsCudaKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu`, the function of that
  name -- a LINE RANGE was written here first and `origin/main` `c2daafb23`
  invalidated it inside this very branch, which is the drift this file keeps
  recording) returns
false for IQ4_NL, Q4_0, Q5_0 and Q8_0, so the MoE seam and this wave's own
quantized projection route either drain to the host or refuse — also another
wave's. The reachability of these four arms from a production entry point is
therefore **VACUOUS, and this wave MEASURED that rather than asserting it**:
mutation M8 below deletes the production call site and both CUDA suites stay
green while `test_qwen4_exp_ple_block` reds. `## Owed` carries the entry. Nothing here
claims a token, a decode or a speed number.

**WHAT DOES CHANGE IS THE SIZE OF THE REMAINING GAP, and it is the reachability
story worth telling.** Three CUDA arms already existed and were unreachable
BEHIND these four — `vt::Qwen4ExpPleConv` and `vt::Qwen4ExpPleGate`
(`cuda_qwen4_exp_ple.cu:371,374`) and `vt::Qwen4ExpGatedResidualWriteBack`
(`cuda_qwen4_exp.cu`, the `RegisterOp(OpId::kQwen4ExpGatedResidualWriteBack, ...)`
line -- it was `:194` before this wave added the mixer to the same Registrar and
pushed it down, a stale anchor created INSIDE its own change). Each sits
IMMEDIATELY AFTER a CPU-only op in the same
function: `qwen4_exp_ple_block.cpp:489` and `:499` are `vt::RmsNormGroup` calls
that precede the gate at `:531` and the conv at `:580`; `qwen4_exp_forward.cpp:418`
and `:476` are mixer calls that precede the write-back at `:457` and `:506`. So
this wave converts seven kernels from "cannot run" to "blocked by exactly one
remaining op", where before it was four separate blockers.

**`vt::RmsNormGroup` UNBLOCKS NO OTHER MODEL, and the dispatch brief's claim that
it did is corrected here.** It is a shared `vt::` seam and any model may call it,
but the tree has exactly three non-test call sites and all three are `qwen4_exp`:
`src/vllm/model_executor/models/qwen4_exp_ple_block.cpp:489, :499, :538`
(`grep -rn 'RmsNormGroup(' src/ --include='*.cpp' --include='*.h' --include='*.cu'`,
excluding the declaration, the CPU kernel, the fn typedef and the args struct),
and `qwen4_exp_ple_block.h:11` says so in its own words: "Callers outside its own
translation unit: zero." It was done first because it is the smallest of the
four, not because it carries leverage.

#### The four decisions, made

| op | the decision W6-CUDA recorded | this wave's answer | gate |
|---|---|---|---|
| `vt::RmsNormGroup` | the reduction WIDTH, and whether it unifies with the mixer's | **f32, ascending, sequential, ONE THREAD PER GROUP.** It does NOT unify: the mixer's `double` is that op's contract and this op's f32 is the width its goldens were dumped in | **byte-identical**, a byte comparison |
| `vt::Qwen4ExpQsaCompress` | the width, plus the `round_intermediates_to_bf16` arm | **f32 ascending for the pool (per `d`, in parallel) and for the sum of squares (on thread 0, sequential).** The bf16 round trip is implemented op for op with `__float2bfloat16`, which is round-to-nearest-even exactly as the host `F32ToBF16` is | **byte-identical in BOTH flag arms**, a byte comparison |
| `vt::Qwen4ExpQsaGatherAttention` | the VISIT ORDER, a DEVICE-side `keys_visited`, and gather-vs-mask | **all four of the CPU arm's orders preserved** (below). `keys_visited` is counted AT THE READ in a per-thread counter, warp-reduced and `atomicAdd`ed to a device slot copied back only when the caller passed the instrument. The expansion stays an ADDRESS walk | one f32 ulp; the only divergence source is `expf` |
| `vt::Qwen4ExpGatedResidual` | the reduction WIDTH: a 571x separation from f32 at group size 2560 | **`double`, inherited from the CPU arm rather than chosen**, narrowing exactly where the host narrows. The three PROJECTIONS go through the shared `vt::MatmulBT` seam | the ORACLE's own 1e-5 band — see below |

#### Why three of the four are byte-identical and the mixer is not

`-ffp-contract=off` is pinned for the host provider (CMakeLists.txt:41-56) and
nvcc's `-fmad` is ON and unpinned, so every multiply-add that must match the host
is spelled `__fmul_rn` / `__fadd_rn` — the measure `cuda_conv1d_general.cu:138-141`
and `cuda_qwen4_exp.cu` already take. The divides are `__fdiv_rn` / `__frcp_rn` /
`__ddiv_rn` and the roots are `sqrtf`; IEEE-754 requires both correctly rounded,
nvcc's defaults (`-prec-div=true`, `-prec-sqrt=true`) keep them so, and glibc
gives the same guarantee, so those agree by the standard rather than by an
intrinsic. With the reduction ORDER held identical there is then no operation
left that CAN differ.

**AND THAT MAKES THE TWO BYTE-IDENTICAL ARMS' ORACLE GATES TRANSITIVE, EXACTLY.**
The CPU arms are held to the transformers goldens by
`tests/vt/test_ops_rms_norm_group.cpp` and
`tests/vllm/models/test_qwen4_exp_qsa_device.cpp`; the CUDA arms are held to the
CPU arms by equality; equality composes. That argument is available only BECAUSE
the relation is equality, and it is stated rather than assumed.

`vt::Qwen4ExpQsaGatherAttention` cannot reach it, and exactly one function is
why: `exp`. The CPU arm calls `std::exp` on a float, the device arm calls `expf`,
CUDA documents up to 2 ulp for `expf` and glibc's is correctly rounded. The gate
MEASURES that difference and prints it on a green run rather than holding a bound
chosen to pass.

**THE MIXER GIVES UP MORE, AND THE TRADE IS DELIBERATE.** Its CPU arm's
`ProjectRow` routes a block-quantized weight to `vt::MatmulBTQuant` and a float
weight to a PRIVATE `LinearNoBias` that accumulates `sum_i w[o*K+i]*x[i]` in f32
in index order — `LinearNoBias` exists there precisely so the float arm stays
bit-identical to the pre-W5p kernel. A device `LinearNoBias` would reproduce that
order and hand this arm a byte gate. It would also be a hand-written GEMV beside
`vt::MatmulBT`, which is the parallel path AGENTS.md "Shared seams" forbids, and
it would give the released checkpoint's 194 Q8_0 mix weights a second private
route on the one device where the quantized GEMM lives. So both float and block
weights go to `vt::MatmulBT`, which dispatches a block dtype to
`kMatmulBTQuant` itself, and the CONSEQUENCE is recorded rather than hidden: a
device GEMM re-associates the K reduction, so this arm is NOT bit-identical to
its CPU sibling. Its bound is the ORACLE's 1e-5 and not one ulp, because both
arms are held to the golden at 1e-5 and their mutual difference cannot usefully
be asserted tighter than the band they each live in. Two divergence sources, and
no third: the GEMM's association and `exp`.

#### The four orders the gather arm preserves, and the speed it forfeits

The CPU arm's header calls the ascending visit order "load-bearing, not
cosmetic: a gather's visit order IS the softmax's reduction order, and ascending
is what makes a sub-budget gather reduce over exactly the dense sequence and so
be BIT-identical to dense attention rather than merely close". This arm keeps all
four:

1. the **dot** `sum_d q[d]*k[d]` is sequential ascending f32 on ONE thread,
   never tree-reduced;
2. the **max** over the gathered rows is split across threads and block-reduced,
   which is exact and order-free — `max` is associative and commutative in IEEE
   arithmetic and it is the one reduction here that may be parallelised;
3. the **denominator** `sum_p w` is accumulated ascending by one thread;
4. the **value accumulation** `acc[d] += w * v[d]` runs ascending over `p` and is
   parallel across `d` only, which each thread owns alone.

A 32-row tile carries thread 0's weights to the block so the barrier count is
`|sel|/32` pairs rather than `|sel|`, and it changes no order: thread 0 still
walks `s` ascending, still accumulates `denom` ascending, and the block still
applies each tile in ascending `s`.

**THE COST IS STATED RATHER THAN LEFT TO BE FOUND.** Pass 2's dot is sequential
on thread 0, so a block spends `|sel| * head_dim` dependent f32 operations on one
lane — ~525k per (token, head) at the released config. The lever that removes it
is named and NOT taken here: a deterministic tree reduction over `d` would
PRESERVE the gather-vs-dense property, because the dot's order would then depend
on `head_dim` alone and be identical in the sub-budget and the dense run, while
breaking the CPU-vs-CUDA relation the sequential dot keeps. That trade is
declined on the run that first puts this kernel on a device, because it would
replace a measured `max|diff|` with an argued one. Recorded under `## Owed` with
its condition: take it after a red-first measurement that shows the sequential
dot is the bottleneck.

#### Three refusals these arms impose that their CPU arms do not

- **`head_dim > 1024` is refused BY NAME** by both QSA arms. One thread block
  serves one (token, head) or one compressed block and each thread owns one
  element of the head dim, so the block cannot be wider than the device maximum.
  `head_dim` is 256 and `index_head_dim` 128 in the released config. A refusal
  that names the limit is correct where a silent wrap would be a truncated row.
- **A MALFORMED SELECTION POISONS THE ROW WITH NaN** rather than being refused by
  name. The CPU gather `VT_CHECK`s that each block id is ASCENDING and inside the
  complete-block count, that `kv_lens[t]` fits the cache, and that a paged entry
  names a physical page the cache holds. A device kernel cannot throw, and
  `block_ids` / `kv_lens` / `kv_block_table` are DEVICE-resident on a CUDA queue
  so the host dispatcher cannot read them either — the position
  `cuda_qwen4_exp_ple.cu` already takes for `query_start_loc`, whose wrapper
  checks run `if (q.device.type == kCPU)` only. This arm makes those
  preconditions the caller's AND refuses to read out of range: a violating
  (token, head) reads NOTHING out of bounds and writes NaN across its whole
  output row, leaving every other row untouched. NaN over a clamp or a skip,
  because both of those return a plausible tensor, and `MaxAbsDiff` returns
  +infinity on any non-finite operand (#449) so a poisoned row cannot read as a
  match in any gate in this tree. The gate carries the case.
- **The mixer takes ONE `cudaMalloc` per call** for its `normed`, `low`,
  `gate` and `inject_pre` intermediates, freed through a scope guard because
  every `vt::` call between them can throw. The CPU arm holds the same buffers in
  `std::vector`s; a device block cannot, because `hc*hidden` is thousands of
  floats at the released config. A per-call allocation in a decode loop is a cost
  and it is recorded under `## Owed` as a SPEED item rather than papered over
  with a static cache that would not be re-entrant.

#### `keys_visited` is a device counter now, and that debt is discharged

`Qwen4ExpQsaAttnArgs::keys_visited` says in its own words: "A host pointer, on
the `GdnArgs::query_start_loc_host` precedent; a CUDA arm owes a device-side
counter and its copy-back." This is that. It is INCREMENTED AT THE READ in a
per-thread counter, warp-reduced and `atomicAdd`ed once per warp, never assigned
from the selection — the distinction the args comment spends a paragraph on,
because W4's first fresh review found a counter set from `sel.size()` under which
a dense walk still reported the sparse figure and passed 12 cases. The device
slot is allocated, zeroed, read back and freed ONLY when the caller passed the
instrument, so the production path takes no allocation and no stream
synchronisation.

**AND THE COUNTER ALONE IS STILL NOT A SET EQUALITY, which is why the gate
carries two instruments.** W5b-4 mutation M11 proved a dense masked walk can
report the sparse number, because a mask-shaped port changes the loop and the
counter together. The gate therefore ALSO runs the CUDA arm over a cache whose
rows no query ever selects are NaN: a gather never addresses them, a mask
multiplies them by a zero weight into `0.0f * NaN`, and the answer must come back
BITWISE equal to the clean run. Membership gives "read set is a subset of
selected"; the counter gives "|read set| == |selected|"; together they are set
equality, and the margin between the sparse and the dense count is printed.

#### Evidence, and the exact boundary of what it covers

**THE DEVICE IS `thor:gpu0`**, a Jetson Thor at **`sm_110`**, 2026-08-31, `rc` job
`5d2cd023`, inside a lease (`rc run -d thor:gpu0`). Toolkit `nvcc` 13.0.88,
driver 595.78, configure
`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`.
The gated tree is `12071a5dd`, which already carries the merge of `origin/main`
`c2daafb23`, so this is a gate on the MERGE RESULT and not on the branch as it
stood. **EVERY rc BELOW WAS PRINTED BY THE JOB AND READ, never derived** — the
one soft spot W6-CUDA had to label a derivation.

`cuobjdump` confirms the objects are genuinely built for this architecture, which
no rc can show on its own: `cuda_qwen4_exp_qsa.cu.1.sm_110.cubin` and
`cuda_rms_norm_group.cu.1.sm_110.cubin`, 2 of 2.

**THE RED IS A REAL ABSENCE, NOT A MUTATION.** The red leg builds a tree that is
`12071a5dd` with the two new translation units DELETED and `cuda_qwen4_exp.cu`
plus `CMakeLists.txt` taken from `9fb40279d`, so the four registrations simply do
not exist while every test does.

```text
### CONFIGURE RC=0
### RED BUILD RC=0     ### GREEN BUILD RC=0     ### FINAL BUILD RC=0
### red   test_ops_rms_norm_group_cuda RC=1     ### green ... RC=0   ### final ... RC=0
### red   test_qwen4_exp_cuda_reductions RC=1   ### green ... RC=0   ### final ... RC=0
### red   test_qwen4_exp_cuda RC=1              ### green ... RC=0   ### final ... RC=0
### RESTORE CHECK RC=0
```

The red FAILS FOR THE INTENDED REASON and the message says so verbatim:

```text
CHECK_NOTHROW( vt::GetOp(vt::OpId::kRmsNormGroup, DeviceType::kCUDA) ) THREW:
"vt: no kernel for op RmsNormGroup (id 140) on device cuda (type 1), and the
 portable CPU reference tier is NOT eligible: this backend does not report its
 device memory host-addressable ..."
```

Red counts: `test_ops_rms_norm_group_cuda` 5 of 7 cases failed;
`test_qwen4_exp_cuda_reductions` 10 of 10; `test_qwen4_exp_cuda` 1 case and
exactly 4 assertions, which are the four flipped registration checks. Green
counts: **7 cases / 70 assertions**, **10 cases / 120 assertions**, and
**12 cases / 351 assertions**, all `SUCCESS!`. The last number is unchanged from
W6-CUDA's own `sm_121a` run, so flipping its four assertions moved nothing else.

##### What the arms actually measure

| claim | measured |
|---|---|
| `vt::RmsNormGroup` CUDA == CPU, **bitwise**, all 18 dtype triples | **0 bytes differ** in every one of the 18 |
| `vt::RmsNormGroup` CUDA vs the transformers golden | max\|diff\| 1.19e-07, 5.96e-08, **0**, **0** (cases A-D) against 1e-05 |
| `vt::Qwen4ExpQsaCompress` CUDA == CPU, **bitwise**, both `round_intermediates_to_bf16` arms | **0/32 and 0/80 differ**, max\|diff\| 0 |
| `vt::Qwen4ExpQsaCompress` CUDA vs the golden | relL2 **exactly 0** on both cases against 1e-06 |
| `vt::Qwen4ExpQsaGatherAttention` CUDA vs the golden | relL2 7.90e-08 and 8.95e-08 against 2e-03 |
| ... CUDA vs CPU, bound ONE f32 ulp relative | max\|diff\| **2.384e-07** against 3.73e-07 and 2.56e-07; 386/1408 and 1037/2944 elements not bitwise equal |
| ... a sub-budget gather is BIT-IDENTICAL to dense | **0/1408 differ** |
| ... the PAGED arm equals the contiguous one | **0/2944 differ** |
| ... `keys_visited` vs a count derived from the HOST expansion | **1440 == 1440**, dense 2208, **margin 768** |
| ... the read SET: rows this query does not select set to NaN | 12 of query 22's 23 visible rows poisoned, **0/128 differ** |
| ... a malformed selection poisons its row | **128/128 outputs NaN**, and no other row touched |
| `vt::Qwen4ExpGatedResidual` CUDA vs the golden | max\|diff\| 4.47e-08 to 1.19e-07 against 1e-05 |
| ... CUDA vs CPU, bound the oracle's own band | max\|diff\| 5.96e-08 to 1.19e-07; 2-4 elements per case not bitwise equal |
| ... the hyper stream is READ-ONLY | **0 bytes differ** on all four cases |

**THE GATHER'S ARM-VS-ARM MARGIN IS THIN AND IS REPORTED AS SUCH: 2.384e-07
against a 2.560e-07 bound, 93% of it.** That is not a bound chosen to pass — one
f32 ulp relative was written down before the kernel ran, and the divergence has
exactly one named source, `expf`, which CUDA documents at up to 2 ulp where
glibc's is correctly rounded. The compressor beside it, same file, same
intrinsics, no transcendental, comes back at **0**. If this number moves, `exp`
moved; do not widen the bound.

##### The mutation battery, with applied-proof and restore-proof

Every mutation printed a `sha256 before=... after=...` pair proving it changed
the file, and a `RESTORED byte-for-byte` line proving the tree came back. **All
eight applied, all eight built (`BUILD RC=0`), so not one is a compiler proof
wearing a test verdict**, and `RESTORE CHECK RC=0` at the end verifies all four
kernel files against the sha256 manifest taken before the battery.

| mutation | what it breaks | run rc | reading |
|---|---|---|---|
| M1 | `RmsNormGroup` reduces over the WHOLE ROW | **1** | RED |
| M2 | the `+ 1` gamma fold dropped (#2218) | **1** | RED |
| M3 | the compressor ropes at the block's LAST token | **1** | RED |
| M4 | the pool window OVERLAPS (DeepSeek-V4's, not QSA's) | **1** | RED |
| M5 | THE MASK-SHAPED PORT: the gather walks all `kv_len` rows | **1** | RED |
| M6 | the mixer's division moves OUTSIDE the SiLU | **1** | RED |
| M7 | the kCUDA registration deleted, kernel kept referenced | **1** | RED |
| M8 | the PRODUCTION call site deleted | see below | the reachability answer |

M1's first draft was a BUILD failure rather than a red, because dividing by `h`
orphaned `gs_f` and `-Werror` refused it. That is the "a mutation that never ran
reads as a passing test" shape, caught by the script's own
`no binary, so no run` branch, and repaired by redefining `gs_f` instead. It is
recorded because the next person writing a mutation here will hit it.

##### M8: the reachability answer, measured on both sides

M8 deletes `vt::RmsNormGroup`'s first PRODUCTION call site — the one inside
`RunQwen4ExpPleBlock` that `ModelRegistry::Forward` reaches through the layer
loop — and runs THREE gates against it:

```text
### MUT M8 test_qwen4_exp_ple_block        RUN RC=1     <- the call site is LIVE
### MUT M8 test_ops_rms_norm_group_cuda    RUN RC=0     <- the CUDA arm survives it
### MUT M8 test_qwen4_exp_cuda_reductions  RUN RC=0     <- so does this one
```

**Both readings are the finding.** The first says a production path really does
reach this op today, on CPU, so the call site is not decorative. The second and
third say **NOTHING IN PRODUCTION REACHES THIS WAVE'S CUDA ARMS**: a gate that
survives the deletion of the production call site measures a class, not a
capability. AGENTS.md "Nothing lands dead" requires that be stated rather than
discovered later, and this is it — stated, and MEASURED rather than asserted,
which is the difference from W6-CUDA's own "vacuous" claim.

**THE THREE W6-CUDA ARMS ARE STILL NOT LIVE EITHER, AND THAT IS THE DIRECT
ANSWER TO THE QUESTION THIS WAVE WAS ASKED.** `vt::Qwen4ExpPleConv`,
`vt::Qwen4ExpPleGate` and `vt::Qwen4ExpGatedResidualWriteBack` sit immediately
after this wave's ops in the same two functions, so closing four blockers moved
them from "four ops away from runnable" to "one op away". The remaining op is the
block-decoding n-gram gather: `EmbeddingKernelCuda` still refuses a table that is
not f32 or bf16 by name (`cuda_ops.cu`, "cuda embedding: unsupported table dtype
(f32/bf16 only)"), so `DeviceQuantGatherSupported` still refuses CUDA and
`ModelRegistry::Forward` — which is all-or-nothing — cannot run a step on a CUDA
queue. That op is another wave's file territory and is deliberately untouched
here. **Seven CUDA kernels are now blocked by exactly one thing where they were
blocked by five.**

#### The fresh review, and the two blocking findings it returned

A fresh reviewer ran the gather kernel on `thor:gpu0` at five shapes, read the
byte-identity arithmetic line by line against the CPU arms, reproduced the
`0c8067e054b3eca3` oracle hash and reconstructed the 70-assertion count
independently. **It found the four kernels correct and every finding was about
the GATE.** That split is the useful part of the result: the arms compute the
right answers, and the suite that said so could not have noticed if they did not.

**F1, BLOCKING — the arm-vs-arm bound was a tolerance fitted to one fixture.**
`kUlpTol = 1.20e-7` failed on a CORRECT kernel at every shape the reviewer
measured, and worse as the selection grew: 112% of its own budget at the suite
geometry, 186% at `|sel| = 202`, **290% at the released `|sel| = 2050`**, 121% at
9000 pairs. Two errors compounded. One ulp was the wrong constant, because CUDA
documents `expf` at TWO. And the bound scaled the wrong way: it was proportional
to `max|out|`, and `out` is a weighted average whose magnitude shrinks toward the
mean as the selection grows, while the error is governed by `max_p |v_p - out|`,
a spread that does not shrink. Perturbing `exp` by ±1 ulp -- half what CUDA
permits itself -- already took the old bound to 140% of itself.

It is replaced by a bound DERIVED per output element from each fixture's own
inputs, `kExpRelDiff * max_p|v_p[d] - out[d]| + 2|sel| u max_p|v_p[d]|`, with
`kExpRelDiff = 2.5 * 2^-23` from CUDA's 2 ulp plus glibc's correctly-rounded
0.5. The derivation is written out beside the constants. The gate now prints the
scale-free actual/bound RATIO, which is what a reader should watch: it is
invariant to the fixture, so a change in the kernel or in the toolchain's `expf`
moves it visibly while the bound follows the data.

**F2, BLOCKING — every loop that exists for scale ran exactly one iteration.**
`|sel|` was at most 11 against a 32-row tile; `T*HQ` at most 92 against a
4096-block grid; the compressor's `nb` was 2 and 5; the norm's pair count 12 and
259. The reviewer proved this was not hypothetical with two device mutations:
dropping the gather's cross-tile denominator carry and collapsing its grid stride
are each BYTE-FOR-BYTE identical to the unmutated kernel at those shapes, and the
first is 18.2 / 352.6 / 568.2 at 7 / 33 / 65 tiles. Five fixtures now cross those
boundaries -- 37, 64 and exactly-2-plus-tail gather tiles, 5200 gather pairs,
4100 compressor blocks, 1,075,200 norm pairs -- and each asserts its own TAIL IS
LIVE, so a collapsed loop cannot pass by two zeros agreeing. M9 to M12 are the
mutations, added as standing acceptance criteria.

**F3 — the "sub-budget is BIT-IDENTICAL to dense" case was vacuous.** On
`kSubBudget` every query has `complete <= topk`, so `DsaTopkSelect` takes its
all-select branch and the `dense` buffer the case built was the SAME buffer; it
measured determinism, which this file's own comment already called insufficient,
and its vacuity guard was an `INFO`. Running it on `kOverBudget` does not repair
it either: above the budget the selection is a strict SUBSET and must not equal a
dense walk. The non-vacuous form is INVARIANCE -- two selections naming the same
rows through different `topk` widths must agree to the bit -- with the guard
promoted to `CHECK` and a `REQUIRE` that the row sets really are equal.

**F4 — the mixer's block-dtype branch was never entered on a device**, though
the kernel's header argues for `vt::MatmulBT` precisely because the released file
carries 194 Q8_0 mix weights. A Q8_0 case now enters it, and the device settled a
question the case had guessed at: **it does not refuse.**
`IsCudaKeepQuantSupported` returning false for Q8_0 does not stop the branch
running on CUDA.

#### Two lessons the device run taught, both about the gate rather than the code

**A BOUND DERIVED FOR ONE ROUTE MUST NOT BE APPLIED TO A DIFFERENT ROUTE, and
this wave made that mistake TWICE.** F1 was the first. The second was inside F4's
own repair: the new Q8_0 case asserted `Q8_0 vs f32 on CUDA` against the mixer's
1e-05 oracle band, reasoning that the weights are exact in Q8_0 so both arms
compute the same function. They do not: `kMatmulBTQuant` also QUANTIZES THE
ACTIVATION, and `normed` is not exact in Q8_0. The device measured 7.549e-05 --
a real difference held to a bound derived for something else, exactly the shape
F1 had just been repaired for. The comparison that tests THIS wave's kernel is
the same route on both devices, CUDA Q8_0 against CPU Q8_0, and the f32 gap is
now printed as the activation-quantization measurement it is rather than
asserted.

**A MUTATION THAT FAILS TO BUILD IS NOT A VERDICT, AND nvcc's `-Werror=all-warnings`
MAKES THAT EASY TO HIT.** M12's first spelling replaced `idx += step` with
`idx += total`, which orphaned `step`; nvcc refused it with
`error #177-D: variable "step" was declared but never referenced` and the harness
reported `BUILD FAILED -- no binary, so no run. Not counted as a red` rather than
letting a stale binary answer. M1 had hit the same class earlier. The repair
edits the stride's DEFINITION instead. **The harness guard is what made both
visible**, and it is the reason a mutation battery must delete the binary before
it builds.

**AND A MUTATION KILL IS MEANINGLESS WHEN THE SUITE'S BASELINE IS RED.** The run
that first carried F4 had `green test_qwen4_exp_cuda_reductions RC=1`, so every
mutation targeting that suite -- M3, M4, M5, M9, M10, M11 -- reported `RUN RC=1`
for a reason that had nothing to do with the mutation. Those readings are VOID
and are not counted anywhere in this record; the battery was re-run once the
baseline was green. A red baseline turns a mutation battery into a row of
uninformative ones, which is the mirror image of the stale-binary trap.

**nvcc DOES NOT GET THE PROJECT'S `-ffp-contract=off` PIN**, verified on the
device's own compile line: `add_compile_options` applies it to
`COMPILE_LANGUAGE:CXX` (and HIP, and OBJCXX) and never to CUDA. That is exactly
why these kernels spell every multiply-add `__fmul_rn`/`__fadd_rn` instead of
relying on a flag, and it is why the reviewer's first device numbers were
inflated ~2x until it re-ran under the pin. The suites are `.cpp` and DO get the
pin, so this wave's harness is uncontaminated.

##### The battery RAN TWICE, and the second run is on the pushed tree

The whole sequence was re-run end to end on `thor:gpu0` after the branch was
merged onto `origin/main` again, and it reproduced **every** rc and **every**
measurement: red 5/7, 10/10 and 4 assertions; green and final 7 cases/70
assertions, 10/120 and 12/351; M1-M7 RED; M8's three-way verdict unchanged
(`test_qwen4_exp_ple_block` rc 1, both CUDA suites rc 0); `RESTORE CHECK RC=0`;
`FINAL BUILD RC=0`; 2 of 2 `sm_110` cubins.

**The mutation hashes are byte-identical across the two runs** -- M1
`23dcaf40...` -> `64b518b8...`, M3/M4/M5 all from `740eb5cd...`, M7
`23dcaf40...` -> `38efc584...` -- and so are the floating-point measurements
(`5.96046448e-08`, `2.38418579e-07`, `1.1920929e-07`). Two independent runs on
one device agreeing to the bit is what makes these numbers a measurement rather
than a sample.

**A CAVEAT ON WHICH TREE, AND THE FIRST VERSION OF IT DID NOT CARRY.** The
second run gated `20cd838fd`. The paragraph that stood here argued that the
result transferred to the pushed head because "the NINE files this gate compiles
for this change" were `sha256`-identical between the two trees.

**That argument is invalid and a fresh review caught it.** Nine files is not the
compile closure. `git diff 20cd838fd 6e7d651a7` is SIXTEEN files, and three of
them are `include/vt/dtype.h`, `src/vt/dtype.cpp` and `src/vt/cpu/cpu_ops.cpp`.
All three `.cu` arms `#include "vt/dtype.h"`, and `cpu_ops.cpp` holds the CPU
`RmsNormGroupKernel` this wave's byte gate compares against, so "the nine files
are identical, therefore the result transfers" does not follow. The disclosure
recorded separately -- that `F32ToBF16` is byte-identical, that `BF16ToF32` and
`F16ToF32` differ only by the `AsF32` -> `detail::BitsToF32` rename, and that
`RmsNormGroupKernel` hashes `0c8067e054b3eca3` either side -- is what actually
carries the argument, and it is a SEMANTIC argument, so the risk is low rather
than zero. **The device gate has never run on the pushed head**, and that is the
honest boundary; the run recorded below closes it.

##### A conflict-free merge moved the converters this wave's byte gate rests on

Recorded because it is the exact shape this row keeps meeting and because the
merge raised no conflict. `origin/main` `08fa2f5aa` (row VT-CPU-ELEM-DISPATCH)
rewrote `include/vt/dtype.h`, INLINING `SizeOf`, `F16ToF32` and `BF16ToF32` and
renaming `AsF32` to `detail::BitsToF32`. Both of this wave's byte-identity
claims depend on those functions: the CUDA arms assert `__float2bfloat16` is the
same round-to-nearest-even as `vt::F32ToBF16`, and the suites narrow every
operand through them so that both arms are handed the same bits.

Checked rather than assumed, three ways. `F32ToBF16` is **byte-identical** and
still out of line -- it was not part of that change -- so the RNE equivalence is
untouched. `BF16ToF32` and `F16ToF32` are semantically identical, differing only
by the `AsF32` -> `detail::BitsToF32` rename. And
`cpu_ops.cpp::RmsNormGroupKernel`, which is the ORACLE the CUDA norm is compared
against, hashes the same before and after (`0c8067e054b3eca3`). The tree then
rebuilt clean and all eight neighbouring suites passed
(`test_ops_rms_norm_group` 69, `test_qwen4_exp_qsa_device` 4697,
`test_qwen4_exp_hc_device` 516, `test_qwen4_exp_ple_block` 100,
`test_qwen4_exp_forward` 429).

##### The gate after the review repairs, on the PUSHED head

`thor:gpu0`, `sm_110`, nvcc 13.0.88, `rc` job `00b30ca6`, on `b4158d643` --
**the tree that was pushed, not a predecessor**, which closes the provenance gap
F6 named. Every rc read from the job's own stdout.

```text
### CONFIGURE RC=0   ### RED BUILD RC=0   ### GREEN BUILD RC=0   ### FINAL BUILD RC=0
### red   {rms_norm_group_cuda, qwen4_exp_cuda_reductions, qwen4_exp_cuda} RC=1 1 1
### green {                    same three                          } RC=0 0 0
### final {                    same three                          } RC=0 0 0
### RESTORE CHECK RC=0
```

`test_ops_rms_norm_group_cuda` 8 cases / 73 assertions,
`test_qwen4_exp_cuda_reductions` 14 / 160, `test_qwen4_exp_cuda` 12 / 351, all
`SUCCESS!`.

**`### BUILD ALL RC=0` ON THIS RUN, WHERE EVERY EARLIER ONE READ RC=1**, and the
difference is not this wave. The full-tree build had been failing on
`tests/vllm/models/test_glm_moe_dsa_schedule.cpp:304` ("cannot convert
`vllm::mla::MlaSharedSelection*` to `vt::Tensor*`"), a break `origin/main`
carried and has since repaired; that same error was also what reddened
`build-test-cpu` and `sanitize-cpu` on this pull request. The whole 1,343-target
tree now compiles with these four arms in it, which is a stronger statement than
the named-target builds every earlier run made.

`### CTEST qwen4_exp RC=8` is the five pre-existing failures below and nothing
else.

**THE DERIVED BOUND, AT EVERY SHAPE.** `over = 0` everywhere and the worst
scale-free ratio is **0.2041**:

| shape | \|sel\| | max\|diff\| | derived bound | ratio | over |
|---|---|---|---|---|---|
| `sub_budget` | 2 | 2.384e-07 | 1.337e-06 | 0.178 | 0/1408 |
| `over_budget` | 2 | 1.192e-07 | 5.842e-07 | 0.204 | 0/2944 |
| 1200 rows / 38 tiles | 1200 | 2.980e-08 | 5.723e-04 | 0.000 | 0/128 |
| 2050 rows / 65 tiles | 2050 | 3.166e-08 | 9.777e-04 | 0.000 | 0/128 |
| 2 full tiles + 2-row tail | 66 | 1.192e-07 | 3.176e-05 | 0.004 | 0/384 |
| 5200 pairs / grid stride | 8 | 2.384e-07 | 2.824e-06 | 0.084 | 0/166400 |

The retired constant failed all of these. The ratio falls as `|sel|` grows,
which is the derivation working: the bound now tracks the spread and the
accumulation length instead of `max|out|`.

**THE SCALE FIXTURES DO WHAT THEY WERE ADDED FOR.** `tiles = 38`, `65` and `3`
as the code computes them; `qsa_compress` 4100 blocks byte-identical at
`0/65600`; the grid-stride gather covers 166,400 elements; row-set invariance
holds across `topk` 50 vs 64 over 7 tiles; `keys_visited` 1440 == 1440 against
2208 dense, margin 768.

**F4 SETTLED A QUESTION THE CASE HAD GUESSED AT.** The Q8_0 mix weight does NOT
refuse on CUDA -- `block branch ENTERED, no refusal` -- so
`IsCudaKeepQuantSupported` returning false for Q8_0 does not stop the branch
running. Same route on both devices agrees at **2.98e-08** against the 1e-05
band. The `Q8_0`-vs-`f32` gap is **7.549e-05**, reported and not asserted,
because it measures the ACTIVATION quantization `kMatmulBTQuant` performs and
not this wave's kernel.

**TWELVE MUTATIONS, ALL APPLIED, ALL BUILT, ALL RED, ALL RESTORED
BYTE-FOR-BYTE**, with M12 building this time after its first spelling orphaned
`step` under nvcc's `-Werror=all-warnings`. M9 and M10 are the reviewer's own
MUT-A and MUT-C, which were byte-for-byte invisible before F2's fixtures existed.

**M8, THE REACHABILITY MUTATION, NOW READS ON A GREEN BASELINE FOR BOTH CUDA
SUITES** -- which the earlier run could not claim, because the reductions suite
was red for an unrelated reason and a mutation kill is meaningless against a red
baseline:

```text
### MUT M8 test_qwen4_exp_ple_block        RUN RC=1   <- the production call site is LIVE
### MUT M8 test_ops_rms_norm_group_cuda    RUN RC=0   <- the CUDA arm survives its deletion
### MUT M8 test_qwen4_exp_cuda_reductions  RUN RC=0   <- so does this one
```

##### The neighbouring suites, and the LOAD-TIME refusal that settles reachability

`ctest -R 'qwen4_exp|rms_norm_group'` on the gated tree: **76% passed, 5 failed
of 21**, and the five are `test_qwen4_exp_gguf_load_plan`, `..._gguf_weights`,
`..._layer_loop`, `..._runner`, `..._forward` — **the IDENTICAL set W6-CUDA
recorded as its own pre-existing baseline** on a CUDA-enabled build ("an
IDENTICAL failing set both times"). This wave's three suites and
`test_ops_rms_norm_group` are ABSENT from it, because they passed. **The change
regressed nothing.**

**AND THE REASON THOSE FIVE FAIL IS THIS WAVE'S OWN REACHABILITY ANSWER, in the
tree's words rather than in mine.** `test_qwen4_exp_forward` throws out of
`LoadThroughRegistry` — the production loader — before a single op is
dispatched:

```text
qwen4_exp gguf: `per_layer_token_embd.weight` is the n-gram gather table and it
must stay block-resident, but device 'cuda' has no block-decoding gather kernel,
so the table would expand to bf16 - 95.4 GiB of host memory on the released
checkpoint, which the on-disk device-fit guard (issue #1123) cannot see. The
CUDA gather arm is owed. Load this model with --device cpu.
```

**EVERYTHING FROM HERE TO THE END OF THIS SUBSECTION WAS TRUE WHEN W6-CUDA-B
MEASURED IT AND IS FALSE ON `main` NOW.** KGATHER (#2396, `f937e8063`) landed
after W6-CUDA-B (#2391, `27aaf199a`) and closed the gap this subsection is
about. It is kept rather than deleted because the measurement below is what the
gate actually observed on `thor:gpu0`, and a record that quietly loses its own
superseded reading cannot be audited; the correction rides here instead.

WHAT CHANGED, checked against `main` rather than against the pull request:
`include/vt/ops.h` now has `kEmbeddingQuant` (`grep -c` is **2**, not 0),
`gguf_keep_quant.cpp`'s `DeviceQuantGatherSupported` is now
`return vt::OpRegistered(vt::OpId::kEmbeddingQuant, dev);` rather than
`dev == kCPU`, and `EmbeddingKernelCuda` decodes a block-quantized table instead
of refusing one -- its "unsupported table dtype" string still exists, but it no
longer covers the block case, so citing it as the gather's blocker is
misleading now even though the string is still greppable. **So a `qwen4exp`
GGUF LOADS on a CUDA device today.**

WHAT DID NOT CHANGE, and is the reason this row still claims no GPU run: the
three host reads in `qwen4_exp_qsa_block.cpp` (`CheckRopeLayoutsAgree`,
`IndexerRows`, `Qwen4ExpQsaIndex`) are still on `main`, and
`IsCudaKeepQuantSupported` still excludes IQ4_NL and Q5_0, which the released
UD-IQ1_S uses. Both are live waves' territory. **No token has come out of a GPU
and none is claimed.**

The superseded reading follows, as W6-CUDA-B recorded it:

So on a CUDA build this architecture **did not load**, and the refusal was at
LOAD time rather than at the first forward.

**THE REFUSAL IS CONDITIONAL, AND THE SENTENCE ABOVE SAID "AT ALL" WHERE IT
SHOULD HAVE NAMED THE CONDITION.** `qwen4_exp_weights.cpp` guards it with
`if (!p.ple.layer_ids_zero_based.empty() && !DeviceQuantGatherSupported(device))`.
A `qwen4exp` config naming NO PLE layer has no n-gram table, so nothing gathers
from blocks and nothing is refused. For such a config the load proceeds and
`vt::Qwen4ExpGatedResidual` and both QSA arms WOULD be dispatched on a CUDA
queue; `vt::RmsNormGroup` still would not, because all three of its call sites
are inside `RunQwen4ExpPleBlock`. No published `qwen4exp` checkpoint has that
shape -- the released UD-IQ1_S names PLE layers -- so this is a reachable path
with no artifact behind it rather than a live one. It is recorded because
"does not load at all" is the kind of absolute a reader will rely on. That is a stronger statement than
M8's, and the two agree: M8 says nothing in production reaches these kernels;
this says nothing in production can even construct the model that would. Both are
measured.

At the time of that run `DeviceQuantGatherSupported` returned
`dev == vt::DeviceType::kCPU` and nothing else (`gguf_keep_quant.cpp`),
`EmbeddingKernelCuda` refused a table that is not f32/bf16 by name, and there was
no `kEmbeddingQuant` OpId at all — `grep -c kEmbeddingQuant include/vt/ops.h`
was 0. Three independent readings of one gap. **All three are false on `main`
now**; KGATHER closed that gap and the paragraph opening this subsection carries
the current values. Closing it was another wave's file territory, and that wave
closed it.

## Mutation record — W8CONFIRM (#2031, issue OWED)

W5s answered "does the released artifact emit real tokens on `origin/main`" and
answered it correctly. It did NOT isolate WHICH commit repaired it, and its own
row says it did. This wave supplies the isolation, on one tree, in one job.

**THE GAP, STATED PRECISELY.** W5s compares W7DIAG's probe on `701606e51`
against its own run on `52f7ccbfc`. That base predates W5p **and** W5r, so the
comparison spans two commits and cannot apportion the repair between them. Its
own `VT_CPU_QUANT_REPACK=0` arm cannot close the gap either: with the marker fix
present, BOTH arms are correct, so A and B come back bit-identical and the
comparison has no signal at all. The discriminator prescribed under `## Owed`
("re-run with `VT_CPU_QUANT_REPACK=0`; if the output stops being constant, this
was it") only discriminates on the PRE-FIX tree. Run on the fixed tree it
confirms an invariant; it decides nothing.

**THE SQUARE.** One source tarball (sha256
`64f068d662aa6ea59a889c835584329e8ec94e7abc6f73383a95b42ceb83892d`, `git archive`
of `52f7ccbfc`), one staged copy of the artifact, one compiler, TWO binaries that
differ in exactly the two lines `7a937db8a` added at
`dense_attn_block.h:235-236`:

| Arm | Binary | Env | Completion for `The capital of France is` |
|---|---|---|---|
| `M-ON` | `e18a38a6…` (main) | default, repack ON | `" Paris. Given this fact, what is the capital of France?\n\n<think>\n"` |
| `M-OFF` | `e18a38a6…` (main) | `VT_CPU_QUANT_REPACK=0` | identical to `M-ON` |
| `X-ON` | `cfdf47bd…` (reverted) | default, repack ON | **`"!!!!!!!!!!!!!!!!"`** — 16 tokens of id 0 |
| `X-OFF` | `cfdf47bd…` (reverted) | `VT_CPU_QUANT_REPACK=0` | identical to `M-ON` |

`X-ON` degenerate AND `X-OFF` coherent is the only combination that isolates the
marker drop. Both X arms are the SAME binary and differ by one environment
variable, so the defect requires the repack chain ACTIVE and the markers
DROPPED. That excludes the repack itself, because `X-OFF` runs the defective
binary with the chain off and is bit-identical to `M-ON` at every probe stage.
It also excludes W5p's `IsBlockQuant` fork, which is present in all four arms and
therefore cannot explain a difference between them.

**THE APPLIED-PROOF IS THE LINE COUNT AND THE LINK, NOT THE PATCH RC.** A
mutation that never applied reads as a surviving one. Recorded: `mutation patch
rc=0`, fix lines `2 -> 0` counted in the compiled source, `ninja[mutated] rc=0`
rebuilding 35 of 530 TUs (the `dense_attn_block.h` dependents), and the two
binary sha256s DIFFERENT. The job aborts the causal arms by name if the count is
not 2 then 0, if the patch fails, or if the two hashes match.

**THE STAGE WHERE IT DIES, ON ONE TREE.** `M-ON` and `X-ON` are numerically
identical through the embedding and diverge totally at the first layer:

| Stage | `M-ON` | `X-ON` |
|---|---|---|
| `embed` | `nan=0 l2=0.473868` | `nan=0 l2=0.473868` |
| `stream.after_widen` | `nan=0 l2=0.947736` | `nan=0 l2=0.947736` |
| `stream.after_layer_0` | `nan=0 l2=3.66967` | `nan=51200` (all), `l2=0` |
| `LOGITS` | `zero=0`, argmax `[11751] 15.7873` | `zero=248320`, all `0`, argmax `[0]` |

Every later stage stays all-NaN through `after_layer_47` and
`hidden.after_mixer_collapse`; the Q4_K head quantizes NaN to zero, which is why
248,320 logits tie at exactly `0` and `argmax` returns index 0 = `!`. `X-ON`
reproduces `nan=51200` — the SAME figure W5s cites for the pre-fix tree — on a
binary deliberately mutated for this purpose, which corroborates both runs.

**FOUR MORE PROMPTS, because one prompt cannot separate a fluent model from a
lucky one.** `M-ON` at 16/12/12/12 tokens, greedy:

| Prompt | Completion | `finish_reason` |
|---|---|---|
| `The capital of France is` | `" Paris. Given this fact, what is the capital of France?\n\n<think>\n"` | length |
| `Once upon a time, in a small village` | `", there was a young boy named Tom. Tom was a"` | length |
| `def fibonacci(n):` | `"\n    if n <= 1:\n        return n"` | length |
| `Question: What is the largest planet in our solar system? Answer:` | `" Jupiter."` | **stop** |

Four domains, four correct answers. The code arm is well-formed Python with the
correct base case, and the Jupiter arm ends on the model's own EOS at 3 tokens
rather than the cap — a constant-token path always hits `length` and can never
produce a natural stop.

**MEASUREMENTS.** `thor:gpu0`, worker `rc-worker-n8smh`, aarch64, 14 cores,
`i8mm` present so `QuantRepackActive()` is true. Staging 2448 s for 67.564 GiB
CIFS to worker-local, shard1 sha256 `88a14208…` verified against the pin. Build 1
`rc=0` in 434 s over 530 TUs; build 2 `rc=0` in 61 s over 35. Loads 41 s cold,
then 20 s warm. `VmHWM` 77,684,708 kB (`M-ON`), 76,842,172 kB (`M-OFF`),
77,646,824 kB (`X-ON`), 76,842,100 kB (`X-OFF`). Probe live at 2376 lines
(`M-ON`) and 864 (each other arm); a zero there would have made every reading
VOID.

**WHAT THIS DOES NOT CLAIM.** It is not a token gate: no oracle decoded these
four prompts, and the recorded llama.cpp reading covers only the first one.
Different-but-coherent text was the accepted bar, and this is a 1.6-bit quant.
No speed number: every arm is n=1 on a shared box. UD-IQ1_S only, one sequence,
`--device cpu` only. `logprobs` VALUES were again all `null`, which W5s correctly
attributes to the `step == nullptr` branch of `BuildCompletionLogProbs`; nothing
here reads them.

## DECODEDIV (#2496) — the CUDA decode divergence, and the instrument that names it

**Scope.** One defect: on `thor:gpu0`, over the released
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact through `examples/server`,
greedy, `max_tokens=8`, the CUDA arm emits `11751 271 271 271 271 271 0 0` where
the CPU control on the same tree and the same file emits
`11751 13 15767 411 2029 11 1092 369`. Token 0 agrees. This wave finds the cause
and fixes it. It changes no behaviour the CPU arm has today.

**What the measurement already excludes,** each by measurement rather than by
argument, and each recorded on [#2496](https://github.com/mudler/vllm.cpp/issues/2496):

* the GDN adapter lifetime defect ([#2476](https://github.com/mudler/vllm.cpp/issues/2476),
  fixed by [#2509](https://github.com/mudler/vllm.cpp/issues/2509)) — a freed
  operand returns what the allocator left there, not the launch-blocking arm's
  exact sequence;
* timing, ordering and the whole async-copy class — the production and
  `CUDA_LAUNCH_BLOCKING=1` sequences are byte-identical, and both are identical
  to the original pre-fix measurement on a different tree;
* the GDN QKVZ GEMM extents — every operand extent agrees with its allocation at
  `T=5` and at `T=1`;
* the f32 QSA query buffer ([#2488](https://github.com/mudler/vllm.cpp/issues/2488))
  — no device branch and no prefill/decode fork, so it is present on the CORRECT
  CPU run too;
* the PLE n-gram history read ([#2504](https://github.com/mudler/vllm.cpp/issues/2504))
  — a real defect, fixed and gated, and it moved the output not at all.

**The shape that survives.** A correct token 0 with a bit-stable wrong decode is
state that the second step carries, computed differently on the two arms. The
list of such state is short and this spec can enumerate it, which is what makes
the wave finite: within one step every buffer the forward writes is also read
back by that same step EXCEPT three. The paged K/V and the indexer side cache are
written and then read inside the prefill, so a defect in either is visible in
token 0. The three that are written at prefill and first read at decode are

1. the GDN conv ring and the GDN temporal state (`MambaSpec` states 0 and 1),
2. the PLE conv ring (`MambaSpec` state 2),
3. the PLE n-gram history (`MambaSpec` state 3).

**Method — the instrument comes before the hypothesis.** A whole-output symptom
cannot name a tensor, so the first artifact of this wave is a comparison, not a
fix: `test_qwen4_exp_layer_loop.cpp`'s `#2496` case drives one prefill and one
decode through `ModelRegistry::Forward` on a CPU queue and on a CUDA queue over
one fixture and one pinned pair of token ids, and reports, in order, the prefill
logits, every persistent buffer above, and the decode logits. The FIRST row that
disagrees is the finding. The second step's token is a constant rather than the
first step's argmax, because sampling per arm would feed the two arms different
ids the moment the prefill logits differ at all.

**What the instrument cannot see, stated before it is run.** `qwen4_exp_gguf_fixture.h`
is a miniature whose layer-3 activations sit near 2^18, where one bf16 ULP is
~1024; W5j measured 0 of 192 paged K/V words moving across two different prompts
on it. A CPU/CUDA difference small enough to be absorbed by that store is
invisible here, so a green result is NOT a claim that the device arm decodes
correctly at released width. It is the statement that the difference is not one
this fixture can hold, and the wave then escalates to the released artifact.

**Gates.** The focused gate is the new case plus `test_qwen4_exp_layer_loop`,
`test_qwen4_exp_cuda` and `test_qwen4_exp_cuda_reductions` on `thor:gpu0`
(`sm_110`, the only device in the fleet at this capability), inside an `rc`
lease. The wave's acceptance gate is the released artifact through
`examples/server`: **the GPU must emit `11751 13 15767 411 2029 11 1092 369`**.
Nothing short of that token sequence is a fix, and a green hermetic case is not a
substitute for it.

### Outcome — the first tensor that diverged, and what it named

**The instrument answered on the first run, and it named a tensor rather than a
layer.** `VT_Q4EXP_STATE_FP=1` on `thor:gpu0`, the released UD-IQ1_S artifact,
`--device cuda` against `--device cpu`, one prompt, greedy, `max_tokens=8`. Every
persistent state agrees after the prefill. The FIRST divergence is at **step 1,
the first decode, in the PLE n-gram history**, and it is exact:

| step | `--device cpu` | `--device cuda` |
|---|---|---|
| 0 (prefill, T=5) | `[9338, 369]` | `[9338, 369]` |
| 1 | `[369, 11751]` | `[369, 0]` |
| 2 | `[11751, 13]` | `[0, 0]` |
| 3-7 | rolls the sampled ids | `[0, 0]` |

The history is int64 TOKEN IDS, so this cannot be a rounding difference and needs
no tolerance to read. The FIFO rolled correctly on both arms — `369` moved from
slot 1 to slot 0 — and what the device arm PUSHED was `0` where the host arm
pushed the token that had just been sampled.

**So the defect was never in a state at all.** The history is advanced from
`input_ids`, so a zero pushed means the forward was HANDED token id 0. The cause
is `ModelForwardInput::device_token_ids`: the asynchronous runner's combine
splices each decode row's sampled token into a DEVICE buffer on the main queue
and deliberately leaves the host `token_ids` stale, and this hook read the host
vector. `token_ids_cpu` is zero-initialised, so every decode row embedded and
hashed id 0. TWELVE other translation units already consume that field under
[#1305](https://github.com/mudler/vllm.cpp/issues/1305), which is the same defect
on Qwen3-MoE.

**That explains every constraint the issue had accumulated**, which is the test a
cause has to pass: token 0 is right because a prefill row is not a decode row;
the decode is wrong from the first step because every decode row is; it is
bit-stable across builds and trees because zero is a constant, not a race; and
`CUDA_LAUNCH_BLOCKING=1` cannot move it because nothing here is a launch order.
It also explains why the REPORTED ids stayed plausible — they are the argmax of
each step's logits, and what was broken is the FEEDBACK, not the sampler.

**AND IT CORRECTS THIS SPEC'S OWN FRAMING.** The scope above enumerates three
write-at-prefill / read-at-decode states and calls that list what makes the wave
finite. The list was right and the conclusion drawn from it was not: the
divergence was in one of those three, and the CAUSE was upstream of all of them,
in an input the forward is handed. An enumeration of state can locate a symptom;
it cannot bound where the symptom comes from.

**What the hermetic case could not have found.** It passes `device_token_ids ==
nullptr` on both arms, because a test builds `ModelForwardInput` by hand. It was
still the right first instrument — it costs no GPU and it would have convicted any
of the three states — but the defect lives in a field only the runner sets, which
is why the released-artifact fingerprint is the arm that answered.

**Stop conditions.** Stop and report rather than widen scope if: the hermetic
comparison is green AND the released-artifact run still diverges (the fixture
cannot hold the difference — report that as the finding and escalate to a
per-layer tap on the real artifact); the divergence is in a `vt::` op with an
existing CPU-vs-CUDA gate that the defect passes (the gate's coverage is the
defect, and widening it is its own change); or `thor:gpu0` is out of the pool,
in which case the sm_110 axis is UNMEASURED and says so.

## Wave PREFILLDIV — the CUDA prefill diverges from the CPU prefill (#2547)

### What this wave is for

[#2496](https://github.com/mudler/vllm.cpp/issues/2496) is fixed -- its fix
LANDED as [#2550](https://github.com/mudler/vllm.cpp/pull/2550) at `bb78d1ee8`,
so this wave's measurement tree is now the composition `main` carries -- and the
CUDA arm is fluent. It is not token-exact:

| arm | token ids |
|---|---|
| `--device cpu` | `11751 13 15767 411 2029 11 1092 369` |
| `--device cuda` | `11751 13 15767 411 1928 11 628 567` |

Five of eight agree and the first disagreement is at token 4.
[#2547](https://github.com/mudler/vllm.cpp/issues/2547) measured that the
difference is NOT carried decode state: the PLE n-gram history rolls identically
on both arms, and the model's own output at **step 0** already differs —
`sumabs 28054.1` on CPU against `27964.7` on CUDA over `n=12800`, about 0.3%.
The divergence is therefore in the PREFILL, before any decode state is read.

**"Prefill is right" was never measured.** It rested on token 0 agreeing, and
` Paris` after `The capital of France is` is not a near-tie, so a single argmax
survived a divergence that was already there. This wave replaces the argmax with
a per-tap fingerprint.

### Scope

IN: a per-layer, per-tap CPU-vs-CUDA fingerprint of the `qwen4_exp` forward; the
one arithmetic divergence it names; the fix for that divergence; the
red-then-green transcript through `ModelRegistry::Forward`.

OUT: speed of any arm, `num_reqs > 1`, the positional arm's second step, and
every `## Owed` entry this row already carries. A second divergence the
instrument finds AFTER the first one is fixed belongs to a following wave and to
its own issue.

### Design — the instrument, and why the layer axis is the one that was missing

`VT_Q4EXP_STATE_FP` (#2496) prints the layer loop's OUTPUT once per step. It has
no LAYER axis, so it can say that the two arms disagree and not where. This wave
adds `VT_Q4EXP_LAYER_FP=<steps>`, which prints one line per tap per layer for the
first `<steps>` forward calls:

```
q4fp step=0 L07 tag=blk        dtype=bf16 dev=1 n=12800 nonfinite=0 maxabs=... sumabs=... v=...
```

The taps are placed so that a difference can be attributed to ONE op:
`emb`, `wide`, then per layer `in`, `ple`, `ahc.mixed`, `ahc.inj`, `blk`, `s1`,
`mhc.mixed`, `mhc.inj`, `moe`, `s2`, and finally `out`. `layer_types` puts a GDN
layer at 0, the PLE layer at 2 and the first QSA layer at 3, so the first tap
that leaves its own arm-vs-arm band names the op:

| first divergent tag | implicates |
|---|---|
| `emb` / `wide` | the embedding gather or the widen |
| `L00 ahc.mixed` | `vt::Qwen4ExpGatedResidual` — the mixer |
| `L00 blk` | `RunGdnBlockPaged` |
| `L02 ple` | `RunQwen4ExpPleBlock` |
| `L03 blk` | `RunQwen4ExpQsaBlockPaged` |
| any `moe` | `RunQwen4ExpMoeBlock` |

**IT PRINTS THE DTYPE, and that is not decoration.** A token gate cannot see a
dtype and a value gate cannot see a lifetime; both failures are on this row's
record already (#2493, #2476). A tap that reported values alone would pass a
buffer that is f32 on one arm and bf16 on the other while moving twice the bytes.

**IT COUNTS ITS OWN TAPS.** The last line of a fingerprinted step is
`q4fp step=<s> taps=<n>`, so a run that measured nothing is distinguishable from
a run whose taps agreed. A grep that matches nothing is not evidence of absence,
and this row has already paid for an instrument that never ran: the
`q4exp-gdngemm` job looked for `$BLD/vllm-server` while ninja links
`$BLD/examples/vllm-server`, and it measured nothing at rc 0.

**IT SYNCHRONISES, and that is stated rather than hidden.** Each tap copies the
tensor to the host and drains the queue, so this instrument CANNOT see a race.
It is admissible here only because #2547 measured the divergence as bit-stable
across builds, trees and `CUDA_LAUNCH_BLOCKING`, which is a deterministic
arithmetic difference and not a race. A NEW symptom that appears only without the
instrument is a race and belongs to a different method.

### Risks

- **The two arms disagree everywhere at some magnitude.** Every kernel on this
  path is a separate CPU and CUDA implementation, so no tap is bit-identical and
  "the first tap that differs" is not by itself the answer. The reading is the
  first tap where the relative difference JUMPS by orders of magnitude over the
  taps before it.
- **A tolerance that passes its own suite can still be too loose.**
  `test_qwen4_exp_cuda` is green at 351/351; that bounds each op against its own
  band, not the composition against the CPU arm.
- **The instrument perturbs the schedule.** See above: admissible only because
  the symptom is bit-stable.

### Gates

```sh
ctest --test-dir "$BLD" -R 'qwen4_exp' --output-on-failure
```

plus, on `thor:gpu0` (`sm_110`), the released
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact through
`examples/vllm-server` on one binary, both arms, greedy, `max_tokens=8`, prompt
`The capital of France is`, reporting both token id sequences verbatim.

### Evidence required

1. The fingerprint tables from both arms, and the first tap whose relative
   difference jumps.
2. The upstream `file:line` and revision for the arithmetic the fix restores.
3. Red-then-green on a test that enters through `ModelRegistry::Forward` on a
   `--device cuda` queue.
4. A reachability mutation that deletes the production call site and reds.
5. Both token id sequences, and an explicit statement of which ids agree.
6. Every build and gate rc read literally, never derived from a pipe.

### Stop conditions

Stop and report if the instrument shows the first jump inside an op whose repair
moves a shared seam's contract; that needs its own spec and its own issue. Stop
if the released artifact cannot be staged, and report `PENDING` rather than
substituting another checkpoint.

### Outcome

**MEASURED, AND IT FALSIFIES THIS WAVE'S OWN PREMISE.** Full result in
[the evidence file](../../docs/bench-evidence/qwen4exp-cuda-prefill-divergence-20260902.md);
`thor:gpu0`, `sm_110`, one binary
`c3b355deb75efb0071fe6ec21d5067f5e2520722c672487fd67939c363e1ee58`, three arms,
437 taps per step on every one of them.

**The first tensor that differs is decoder layer 0's Gated DeltaNet block
output**, `rel(sum|x|) = 3.525e-04`, produced from an input that is BIT-IDENTICAL
on both arms — `emb`, `wide`, `L00 in`, `L00 ahc.mix` and `L00 ahc.inj` all read
`0.000e+00`.

**That clears the only named candidate by measurement.** #2547 named
`vt::Qwen4ExpGatedResidual`'s CUDA arm as the largest non-bit-identical surface
on the path, on the strength of the source's own admission that its device GEMM
re-associates the K reduction. On the released weights at the model's real width
it is bit-identical to its CPU sibling at both hyper-connection sites of layer 0.
The re-association is absorbed by the bf16 store. Reading a source comment is not
a measurement, and this is the second time on this row that a candidate named
that way did not survive one.

**The mechanism is the chunked prefill decomposition, and the proof is a
same-binary A/B.** `VT_GDN_CHUNKED=0` routes the CUDA arm to `GdnScanCuda`, the
same sequential recurrence the CPU arm runs, and `L00 blk` falls to `1.062e-06`
— **332x**. Layer 0 is the only layer that can carry this proof, because it is
the only one whose input is bit-identical on both CUDA arms; every later `blk`
measures propagation.

**AND IT IS NOT A DEFECT.** vLLM's vendored Flash-Linear-Attention kernel keeps
`h`, `u`, `w` and `v_new` in bf16 with fp32 accumulation, `A` in fp32 for the
triangular solve, and `final_state` in fp32 — and our chunked port matches all
five, anchor by anchor (evidence §4, vLLM `5559679229`, a forward reference).
The CUDA arm mirrors the oracle. **The CPU arm's exact sequential f32 recurrence
is the arm that does not**, and it is more accurate than vLLM rather than more
correct. Moving CUDA onto the CPU arm would move it away from vLLM.

**The GPU does not emit the control sequence, and the honest count is:**

| arm | ids | agrees |
|---|---|---|
| `--device cpu` | `11751 13 15767 411 2029 11 1092 369` | the control |
| `--device cuda` | `11751 13 15767 411 1928 11 628 567` | 5 of 8 |
| `--device cuda`, `VT_GDN_CHUNKED=0` | `11751 13 15767 264 1103 314 5656 321` | 3 of 8 |

**The sequential arm is 2.8x closer on the hidden state and agrees on FEWER
ids** (`out` rel `1.130e-03` against `3.189e-03`). Token agreement between our
two arms is not monotone in the distance between them, because the decode is an
argmax over near-ties. A CPU-vs-CUDA token-exactness gate is therefore not well
posed for this architecture at this precision, and this wave does not propose
one.

**What is still open is the SECOND source.** Layer 0 separates it: with the GDN
contribution removed, `L00 moe` still differs by `7.269e-05` from an input
differing by `2.1e-05`, and `L00 s.mlp` barely moves (`7.160e-05` ->
`6.815e-05`). The MoE block is the only remaining candidate for an actual defect
on this path. It is named here and NOT diagnosed; the discriminating instrument
it needs is a tap on the router's SELECTED EXPERT IDS asserting set equality and
printing the margin, because a bf16 router logit over this model's expert count
can flip a discrete selection and a flip is a large local change rather than a
rounding one -- and a value-only tap can never tell that from re-association in
the grouped keep-quant GEMM.
[#2552](https://github.com/mudler/vllm.cpp/issues/2552) owns it.

**What landed.** The instrument and its gate, and nothing else. There is no
product-behaviour change in this wave, because the measurement says there is no
defect at the first divergence to change.


## Wave MOEDIV — is the MoE prefill divergence a SELECTION FLIP? (#2552)

### The question, and why the existing instrument cannot answer it

Wave PREFILLDIV ([#2547](https://github.com/mudler/vllm.cpp/issues/2547),
[#2554](https://github.com/mudler/vllm.cpp/pull/2554)) isolated TWO sources of
CPU/CUDA prefill divergence at decoder layer 0, the only layer whose input is
bit-identical on both CUDA arms. The first is the chunked Gated DeltaNet
prefill, and it is not a defect: it mirrors vLLM's vendored Flash-Linear-Attention
precision map at all five anchors. The second is the MoE block, which turns a
`2.1e-05` input difference into a `7.269e-05` output difference and which
`VT_GDN_CHUNKED=0` moves by only 1.7x, against 332x on the Gated DeltaNet tap.
[#2552](https://github.com/mudler/vllm.cpp/issues/2552) owns that second source.

`VT_Q4EXP_LAYER_FP` taps VALUES. **A value tap cannot separate an
expert-selection flip from re-association inside the expert GEMM**, because a
discrete selection has bimodal error and not a tolerance: one token's top-k set
either changes, and that token's MoE output changes by an O(1) amount, or it
does not change at all and the residue is rounding. Averaging both into one
`rel(sum|x|)` destroys exactly the bit that decides which of the two happened.

### What this wave lands

`VT_MOE_SEL_FP=<calls>` — a **selected-expert-id tap** on the reference arm of
the shared sparse-MoE block (`MoeBlock`, `qwen3_5.cpp`), which is the arm
`RunQwen4ExpMoeBlock` reaches on a stacked keep-quant checkpoint. For the first
`<calls>` block invocations of the process it prints, per token:

* the selected expert ids **SORTED**, because the assertion between two arms is
  SET equality and the selection ORDER is not part of it (`vt::MoeCombine` sums
  over the k slots and is order-invariant given the weights);
* `lo`, the smallest selected router logit, and `hi`, the largest rejected one,
  each with its raw bf16 bit pattern;
* `margin = lo - hi` in **LOGIT** space rather than probability space, because
  the softmax denominator is itself a device-order f32 reduction and differs
  between the arms, while the bf16 logits are the values the selection is
  actually a function of;
* `ulps`, the number of representable **bf16** steps between `hi` and `lo`
  under the sign-magnitude total order. `ulps=0` means the two logits are the
  SAME bf16 value and the boundary was decided by the lowest-index tie-break —
  a knife edge that one ulp of re-association anywhere upstream will flip.
  `ulps` is the honest margin for a bimodal error: a probability difference
  reads as "small" for a gap of one representable step and for a gap of fifty.

and, per call, one digest line carrying `sel` (an FNV-1a hash over every token's
sorted id list, so **selection-set equality between two arms is one string
comparison per layer**), the minimum `ulps` over the call's tokens, and four
`sum|x|` axes that decompose the block's own output: `x` (the block input),
`logit` (the router logits), `exp` (the assembled per-slot expert outputs before
the combine) and `shr` (the shared expert). With the selection sets equal, those
four say WHICH of the block's GEMMs carries the residue; with them unequal, the
per-token lines say which token flipped and how near the tie was.

### The counted property

`lines=` on every digest line is the running total of value lines the tap has
printed. **An instrument that never ran and two arms whose taps agreed look the
same in a diff**, and this row has already paid for that twice (a job that
searched `$BLD/vllm-server` while ninja links `examples/vllm-server`, and a
doctest case whose name contained a comma so `-tc` split it). The measuring job
asserts `lines == calls * T` and refuses to report a comparison otherwise.

### Scope, and what this wave does NOT do

It is a diagnostic tap and one measurement. It changes no math on any arm: the
tap reads host buffers the reference path had already downloaded (`h`, `logits`,
`ids`, `expert_out`) plus one guarded download of the shared-expert output, and
it is inert with the environment variable unset. It lands on the reference arm
only. The three fused CUDA arms (`MoeBlockFusedCuda`, `MoeBlockFusedMarlinCuda`,
`MoeBlockBf16Cuda`) keep their ids on device and are NOT tapped, which is
recorded rather than hidden: no `qwen4_exp` checkpoint reaches them, and adding
a device readback to a capturable path to instrument a model that cannot enter
it would be dead code by construction.

### Gates

* `ctest -R test_qwen4_exp_moe` at rc 0, with the tap's ids and `ulps` gated
  against `MoeReference`, the suite's independent double-precision oracle
  reimplementation, on both the bf16 and the keep-quant arm.
* The reachability mutation deletes the tap's production call site inside
  `MoeBlock` and the tap case must RED.
* `scripts/agent-preflight.sh --fail-on-skip` at rc 0.

### Evidence required

One `rc` lease on `thor:gpu0`, one binary, the released
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact (shard 1 sha256
`88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`), the same
prompt and server flags PREFILLDIV used, and three arms: `--device cpu` as the
control, `--device cuda`, and `--device cuda` with `VT_GDN_CHUNKED=0` so the
MoE residue is read with the first source removed. The result is the per-layer
`sel` hash comparison plus the `ulps` distribution.

### Stop conditions

Stop and report `PENDING` if the artifact cannot be staged; do not substitute
another checkpoint. Stop if the answer is that the CUDA arm mirrors vLLM and our
CPU arm is the outlier, exactly as PREFILLDIV concluded for the Gated DeltaNet:
say so with the upstream anchors and close #2552 as answered rather than fixed.
**Do not force a fix.** Changing a CUDA arm to agree with our CPU arm, when the
CUDA arm is the one that mirrors the oracle, moves this model AWAY from vLLM.

### Outcome

**MEASURED. The selections DIFFER, and it is NOT a defect.** Full result in
[the evidence file](../../docs/bench-evidence/qwen4exp-moe-selection-20260902.md);
`thor:gpu0` `sm_110`, job `a5bf074b-0f1a-490d-ab51-5d561857ef9e`, one binary
`b3d5d97c86ffa75f6de9e70e7a036f495962c39c8d76d8ed182b9a44c83beacf`, three arms,
the released UD-IQ1_S artifact verified inside the lease.

**The number that answers the row.** At `E = 512` and `top_k = 10` the router's
top-k boundary is an **EXACT bf16 tie at 79 of 240 prefill token-slots (32.9%)**
on the CPU control, and inside one representable bf16 step at 55.8%. At a tie
the selection carries no information: the lowest-index tie-break decides it. The
flip rate IS the tie rate — 75 of 240 slots flip on the `VT_GDN_CHUNKED=0` arm
(31.3%), 78 on the production arm.

**#2552's own headline number is NOT a flip, and this is the half the issue got
wrong.** At layer 0 with the Gated DeltaNet source removed — the arm its table
was taken on — both arms select the same experts for all five tokens. The
`7.269e-05` residue decomposes onto the expert GEMM: `x` 2.139e-05, `logit`
2.378e-05 (the router GEMM does not amplify), `exp` **1.421e-04** (6.6x), `shr`
4.310e-05. The keep-quant grouped GEMM is the amplifier, and
`cuda_quant_dot.cu:2158-2170` already says what it is — the integer core is
bit-identical and only the per-block float scale sum is reassociated, which
llama.cpp's own CPU/CUDA split does too.

**Why it is not a defect.** vLLM routes this model on bf16 logits at 512 experts
as well: `Qwen4ExpSparseMoeBlock` inherits Qwen3Next's plain `ReplicatedLinear`
gate with no `params_dtype`, `moe_runner.py:897-902` runs it as a plain
`F.linear`, and the f32 widening happens inside `topk_softmax` — exactly our
polarity, our widening point and our lowest-index tie-break. Upstream sits on
the same knife edge. Read at `cdefd9d499`, a FORWARD REFERENCE 1566 commits past
the pin, which carries no `qwen4_exp` at all. **This is a source read, not a
measurement: no vLLM process was run on this checkpoint.**

**One thing to WATCH rather than fix.** vLLM owns a fp32-capable router gate,
`GateLinear` (`gate_linear.py:18-33`), a `PluggableLayer` with three of five
tiers emitting fp32 logits. Qwen3Next does not opt in, so it is unreachable for
this architecture today. If upstream ever flips `qwen4_exp` onto it we mirror
that; widening ours first would move this model AWAY from the oracle, which is
what the stop condition above forbids.

**The gate question, settled with a mechanism.** PREFILLDIV observed that token
agreement is not monotone in numerical distance and proposed no CPU-vs-CUDA
token gate. This says WHY: a third of the routing decisions carry zero margin,
so the emitted sequence is a function of tie-break order rather than of
accuracy. **No CPU-vs-CUDA token-exactness gate is well posed for `qwen4_exp`.**
The selection-set agreement rate and the tie-rate histogram ARE well posed and
are now measurable; a gate against vLLM stays the right target and stays OWED,
because vLLM's GGUF support is an out-of-tree plugin and every safetensors arm
of this model exceeds the largest fleet box.

**What this run could NOT prove, and one thing it found by accident.** The
decode half is VOID: measured on `origin/main` `a99b9c69a`, the CUDA arm answers
`11751 271 271 271 271 271 0 0`, not the fluent sequence PREFILLDIV recorded,
because that wave measured a tree carrying
[#2550](https://github.com/mudler/vllm.cpp/pull/2550)'s decode fix and this one
did not. From step 1 the two arms therefore run different token sequences, so
the comparator's 336-of-336 decode flip count is a different-input artifact and
is published only to say so. **#2550 landed at `bb78d1ee8` while this wave was
writing up**, which is why this branch carries a merge of it; the numbers above
were taken BEFORE that merge and are not re-derived here. A decode-phase
selection comparison is now well posed and is NOT claimed by this wave — the
prefill result stands on its own, and re-running the tap on a post-#2550 tree is
the cheap next measurement.

## Wave TIEBREAK — is the CUDA top-k STABLE at an exact tie? (#2586)

### Scope

Wave MOEDIV (#2552) established that a third of this model's routing boundaries
are exact bf16 ties and concluded, **by source read**, that the arms take the
same lowest-index tie-break. This wave tests the part a source read cannot
reach: whether the CUDA implementation REALISES that tie-break, on
bit-identical input, deterministically.

In scope: the `E = 512`, `k = 10` geometry `qwen4_exp` actually routes; the CUDA
block kernel's parallel argmax; within-arm determinism; the CPU reference at the
same geometry. Out of scope: the expert GEMM re-association MOEDIV §4 already
attributed, the bf16-vs-f32 gate-width question MOEDIV §5 already answered
against upstream, and any change to the router's arithmetic.

### The hypothesis, stated so it can fail

**H:** at an exact tie the CUDA parallel argmax can return an index other than
the lowest tied index, because the answer depends on the reduction grouping
(which thread holds which expert) rather than on the values alone.

**H is false iff** the argmax reduction is a reduction over a TOTAL ORDER —
value descending, then index ascending — applied identically at every level. It
is true if any level compares values only, or compares indices with the wrong
polarity, or lets a warp-uniformity or barrier defect leak a stale partial.

### Why the standing gate cannot decide it

`tests/vt/test_ops_moe_grouped.cpp` "CUDA moe_router_topk parallel == serial
byte-for-byte (adversarial)" is a real gate — it compares against
`vt::cuda::MoeRouterTopKSerialCuda`, byte-for-byte, under both
`VT_MOE_ROUTER_WARP` arms. Two limits keep it away from this question:

1. It sweeps `E in {32,64,128,256}` only.
   `MoeRouterWarpValuesPerThread` (`src/vt/cuda/moe_router_warp.h`) returns 0 for
   E = 512, so the geometry this model routes falls through to
   `MoeRouterTopKKernel<Tin,false>` at 2 experts per thread — **a decomposition
   no case in the tree has ever executed.**
2. Its tie patterns are `(e / 4) % 5` and `e < 12 ? 3.0f : 0.0f`. Both keep the
   tied set contiguous and low, so at `kBlock = 256` the tie lives inside warp 0
   and the CROSS-WARP index comparison never sees a tie it has to break.

### Design

Three assertions, in the order that makes a failure interpretable.

**(1) Closed-form expected selection.** Build a row where the answer is known
without reference to either arm: `h < k` experts at strictly higher, distinct
values; a tied set `S` of `m > k - h` experts all carrying the SAME bf16 bit
pattern, scattered across all 256 threads and all 8 warps; every other expert
strictly lower. The correct top-k is the `h` high experts in descending order
followed by **the `k - h` LOWEST indices of `S`**. Assert index-exact equality
against that. This gates the SEMANTICS and does not merely compare the arms, so
it cannot pass by both arms being wrong the same way.

**(2) Arm agreement, as SET equality with a printed margin.** A discrete
selection has bimodal error; a float tolerance on it gates nothing. Compare
sorted id vectors with `==`, and CAPTURE the tied-set size and the boundary
value so a failure says which side moved.

**(3) Determinism.** The same launch repeated `R` times must give byte-identical
indices; and one row's selection must be identical whether it is run alone or as
row `r` of a large batch, which changes the grid without changing the per-block
work.

Each of (1)-(3) runs on the CPU arm, on `vt::MoeRouterTopK`'s CUDA dispatch, and
on `vt::cuda::MoeRouterTopKSerialCuda`, at E in {256, 512, 1024} and both
`VT_MOE_ROUTER_WARP` arms, in f32 and bf16.

### Upstream

vLLM is the oracle and it DEFINES this tie-break rather than leaving it
unspecified, in both of its kernels, at the parity pin `5559679229`:

| anchor | what it says |
|---|---|
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:536-537` | `topkGating`'s butterfly reduction: "We want lower indices to \"win\" in every thread so we break ties this way", `other_max == max && other_expert < expert` |
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:515-517` | its per-thread scan: "only updated if > (not >=)", so the lowest column wins inside a thread too |
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:707-708` | `case 512: LAUNCH_TOPK(512, ...)` — E = 512 IS a registered `topkGating` width upstream, so upstream runs its register-resident kernel at this model's geometry |
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:186,222,225` | the fallback `moeTopK` uses `cub::ArgMax`, whose contract is the same lowest-key-on-tie |
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:465` | "With 0s, the argmax uses index tie-breaking to pick [0,1,2,...,k-1]" — upstream states the resulting behaviour outright |

So "lowest index wins" is upstream's specified behaviour, not an accident of its
implementation, and assertion (1) above is a mirror of it rather than a local
invention.

### Risks

- **The test passes by not running.** doctest reports `assertions: 0` at rc 0 as
  a success, and a case name containing a comma is split by `-tc`. Mitigation:
  no comma in any new case name; the counted-property assertion below.
- **The CUDA arm cannot be compiled on the dev box** (no `nvcc`). Mitigation: the
  CPU arm and the closed-form expectation run locally and in CI; the CUDA arms
  run inside an `rc` lease and the evidence records the build rc.
- **A green with the warp lever in the ambient state says nothing.** Mitigation:
  reuse `vt_test::ScopedMoeRouterWarp` and `REQUIRE` the pinned state, as the
  standing sweep already does.

### Counted property

Each new case CAPTUREs and CHECKs `tied_seen`, the number of rows whose boundary
was an exact tie by construction. It is `T` by construction and 0 if the builder
ever fails to produce a tie, so a case that measured nothing cannot read green.

### Gates

This tree builds ONE binary per suite; there is no `vt_tests` aggregate and no
`vt_ops` ctest label, so the two commands an earlier draft of this section
carried would have run nothing at all. These are the ones that run:

```sh
cmake --build build --target test_moe_router_tie_stability test_ops_moe \
                              test_ops_moe_grouped test_moe_router_warp_map -j 6
for t in test_moe_router_tie_stability test_ops_moe test_ops_moe_grouped \
         test_moe_router_warp_map; do ./build/tests/$t; echo "$t rc=$?"; done
```

**Read the CASE COUNT and the ASSERTION COUNT beside every rc, never the rc
alone, and never the assertion count alone either.** There are THREE states and
the assertion count separates only two of them:

| state | cases | assertions | rc |
|---|---|---|---|
| CPU-only build — the device cases are `#ifdef`'d out | **1** | 488 | 0 |
| CUDA build, no device — they run, print `no CUDA backend registered; skipping`, return | **4** | 488 | 0 |
| CUDA build with a device | **4** | **4652** | 0 |

All three are rc 0 and two of them read 488, so a Gates instruction that names
only the assertion count cannot tell a CPU-only run from a device-less CUDA run.
§Outcome records a lease where the middle row read GREEN with a broken tie-break
compiled in. The permissive `HasCuda()` skip that makes the middle row possible
is a tree-wide shape and is owed under `## Owed`
([#2603](https://github.com/mudler/vllm.cpp/issues/2603)).

### Evidence required

The doctest assertion counts for the new cases on both arms; the build rc and
host for the CUDA run; and, if and only if a defect is found and fixed, the
token sequence on the released UD-IQ1_S artifact before and after.

### Stop conditions

Stop and report if (1) and (3) are clean on device: that is a valid outcome and
it leaves MOEDIV's conclusion standing. Do NOT widen the router's dtype, and do
NOT change the tie-break polarity, to make any arm agree — either move is a
divergence from the oracle.

### Outcome -- the hypothesis is FALSE and the coverage gap was REAL

**H is false. The CUDA top-k realises the lowest-index tie-break it declares.**
Measured on `thor:gpu0` at tree `64617b150`, job
`5f50e5f7-c730-463f-93cc-3293835ed007`, pod `rc-worker-n8smh`, `NVIDIA Thor`,
driver 595.78, cc 11.0, nvcc `cuda_13.0.r13.0`, `sm_110`. `CMAKE rc=0`,
`BUILD rc=0 objects=586`, 41 `*.cu.o`, `libcudart.so.13` and `libcublasLt.so.13`
in the test binary's own `ldd`.

`test_moe_router_tie_stability`: **4 cases, 4652 assertions, 0 failed, rc = 0.**
Closed-form selection, serial-oracle agreement, 279 repeat comparisons and 2313
batch-position comparisons, at E in {256, 512, 1024}, k in {8, 10}, f32 and bf16,
`VT_MOE_ROUTER_WARP` pinned both ways with the pinned state `REQUIRE`d.

So MOEDIV's reading stands. The `qwen4_exp` CPU-vs-CUDA expert flips are
tie-break order under a perturbed input, not an unstable tie-break, and the
session goal's remaining three disagreeing token ids stay attributed to
arithmetic. **No fix was made, because none was needed, and the three ids do not
move.**

**The coverage gap this wave named was real, and the mutation quantifies it.**
Flipping the block kernel's per-thread strided scan to highest-index-wins
(`cuda_moe.cu`, ONE site, asserted `n == 1`, `MUT BUILD rc=0 objects=1`) is
caught by 104 of the new file's assertions and by **ZERO assertions of the
standing sweep**: `test_ops_moe_grouped` read `1907 | 1906 passed | 1 failed`
before AND after, that one failure being the pre-existing #962 Marlin NVFP4
block-size disagreement on this arch, and `test_ops_moe` read `33451 | 33451`
both times. Restoring returned 4652/4652.

| geometry | pattern | `par == r.expect` failures under the mutation |
|---|---|---|
| E = 512 | 3, 5 | 12, 12 |
| E = 1024 | 3, 5 | 12, 12 |
| **E = 256** | any | **0** |

Zero at E = 256 is the whole finding: the compare the mutation breaks does not
exist below `E > kBlock`, so the standing sweep is blind to **that one defect
class -- an inverted PER-THREAD STRIDED compare** -- and to nothing broader.

**Say only that, because the fresh review measured the other two levels and they
are NOT blind spots.** Inverting the warp `__shfl_down_sync` argmax
(`cuda_moe.cu:169`) reds 372 assertions here and **313 of the standing sweep**;
inverting the cross-warp pass (`:186`) reds 188 here and **199 of the standing
sweep**. Only the per-thread scan reds 104 here and **0 there**. The sweep is
therefore adequate for every reduction level that exists at `E <= 256` and
misses exactly the one that does not, which is a narrower claim than "blind to
it" and the only one the mutations support.

**The CPU arm has its own red-first, measured on the authoring host.** CPU-only
build at the branch head: baseline `1 case | 488 assertions | 0 failed | rc 0`;
flipping the greedy argmax's strict `>` to `>=` at ONE site
(`src/vt/cpu/cpu_ops.cpp:2985`, `grep -c` = 1) gives
`1 case | 488 | 380 passed | 108 failed | rc 1`, all 108 of them
`CHECK( ids == r.expect )` and none of them the counted property; restoring
returns `488/488`. 108 is the whole selection population -- `h < k` is
`REQUIRE`d, so each of the 54 rows selects at least one tied member, and `>=`
mis-selects on every row in both dtypes.

**One negative result worth keeping.** The mutant stayed perfectly
DETERMINISTIC -- all 279 repeat and 2313 batch comparisons passed with the wrong
tie-break in place. Repeatability is necessary and never sufficient, and a wave
that had measured only determinism would have called that mutant clean.

**Re-measured at the merged head.** The numbers above were taken at
`64617b150`; the merge with `origin/main` then touched `src/vt/cuda/cuda_exl3.cu`,
which this binary links. A second `thor:gpu0` lease at `d52bc830f` (job
`186b67e9-4e48-43fb-9424-65735964083c`, `BUILD rc=0 objects=586`) reproduces
every one of them: 4652/0 baseline, 4652/104 mutant, 4652/0 restored, 1907/1906/1
and 33451/0 unmoved on the standing suites, and the same 12/12/12/12 with zero at
E = 256. Only `631da56c4` lands after that run and it edits this file alone.

Full record:
[`docs/bench-evidence/qwen4exp-moe-tiebreak-stability-20260902.md`](../../docs/bench-evidence/qwen4exp-moe-tiebreak-stability-20260902.md).


## Now

`ACTIVE`. **THE COUNT IS THE TABLE, AND THIS SENTENCE NO LONGER RESTATES IT.**
The previous revision opened "Nine reviewed waves have landed. Eight of them are
unreached by design and the ninth, W5a, is the only one with a production call
site" over a table that held THIRTEEN rows before W5e-1 and FOURTEEN after it,
while FIVE landed waves had no row at all: W5b-6, W5d-1, W5d-3 and W5d-4, each
carrying a `## Owed` entry that says it "lands UNREACHED", and W5c-2, whose
`## Owed` entry names it under all four "Nothing lands dead" conditions for the
block-table VALUE nothing reads. That is a live self-contradiction of exactly the
shape [#2288](https://github.com/mudler/vllm.cpp/issues/2288) names, inside the
section that exists to reconcile #2288, and W5e-1 made it worse by adding a
FOURTEENTH row under a sentence that said nine. A prose count beside a table it is not derived
from is a drift lock (`## Owed` records the same failure mode for the production
refusal string). The count is therefore DELETED from the prose and the fact moved
into a per-row column, which a reader can total and a wave cannot leave stale by
adding a row without touching a sentence. Every reviewed wave that has landed has
a row here, and every row says whether anything in production reaches it:

| Wave | Lands | Reached? | Issue |
|---|---|---|---|
| W1 | the config layer: `qwen4_exp` resolves, parses and VALIDATES | **yes** — `Qwen4ExpHfConfigFromGguf`, through the `kGgufArchArms` dispatch row | [#1981](https://github.com/mudler/vllm.cpp/issues/1981) |
| W2 | the hashed n-gram index and the PLE dilated depthwise conv | no | [#1987](https://github.com/mudler/vllm.cpp/issues/1987) |
| W3 | the 4-branch gated-residual hyper-connection stream | no | [#1988](https://github.com/mudler/vllm.cpp/issues/1988) |
| W4 | Qwen Sparse Attention with a GATHER consumer | no | [#1991](https://github.com/mudler/vllm.cpp/issues/1991) |
| W6a | IQ4_NL, Q5_0 and a dequantizing gather, so the artifact OPENS | **yes** — the dispatch row and the dequantizing gather the loader takes | [#1989](https://github.com/mudler/vllm.cpp/issues/1989) |
| W5a | the GGUF weight loader | **yes** — the registry's `load_weights` hook | [#2031](https://github.com/mudler/vllm.cpp/issues/2031) |
| W5b-1 | `RunGdnBlockPaged`, the GDN block seam the forward needs cross-TU | no | [#2110](https://github.com/mudler/vllm.cpp/issues/2110) |
| W5b-2 | the gated-residual hyper-connection stream as two `vt::` ops | no | [#2123](https://github.com/mudler/vllm.cpp/issues/2123) |
| W5b-3 | the PLE dilated depthwise causal conv as `vt::Qwen4ExpPleConv` | no | [#2156](https://github.com/mudler/vllm.cpp/issues/2156) |
| W5b-4 | Qwen Sparse Attention as two `vt::` ops, plus the unmapped-tail probe | no | [#2167](https://github.com/mudler/vllm.cpp/issues/2167) |
| W5b-5 | `Qwen4ExpTextAttention` as ONE block, and the indexer composition in `src/` | no | [#2211](https://github.com/mudler/vllm.cpp/issues/2211) |
| W5b-6 | the gamma polarity of W5b-2's two ops, onto the RAW HuggingFace gamma | no | [#2218](https://github.com/mudler/vllm.cpp/issues/2218) |
| W5c-1 | the KV-cache spec: THREE groups | **yes** — `make_kv_cache` | [#2031](https://github.com/mudler/vllm.cpp/issues/2031) |
| W5c-2 | `GPUModelRunner::gather_group_block_tables`, every group's block table | **path yes, VALUE no** — the gather runs on `execute_model`; no forward reads the tables | [#2249](https://github.com/mudler/vllm.cpp/issues/2249) |
| W5d-1 | the grouped RMS norm as `vt::RmsNormGroup` | no | [#2249](https://github.com/mudler/vllm.cpp/issues/2249) |
| W5d-2 | `BuildMropeCosSinHost` loses `static`: ONE mRoPE table builder, cross-TU | **yes, one hop short** — `qwen3_5.cpp` calls it; that caller is not itself routed from a production entry point | [#2249](https://github.com/mudler/vllm.cpp/issues/2249) |
| W5d-3 | the PAGED QSA consumer, `RunQwen4ExpQsaBlockPaged` | no | [#2249](https://github.com/mudler/vllm.cpp/issues/2249) |
| W5d-4 | the MoE weight adapter, `qwen4_exp_moe.{h,cpp}` | no | [#2249](https://github.com/mudler/vllm.cpp/issues/2249) |
| W5e-1 | the PLE GATE as `vt::Qwen4ExpPleGate` — the op #2249 never surveyed | no | [#2336](https://github.com/mudler/vllm.cpp/issues/2336) |
| W5e-2 | `RunQwen4ExpPleBlock`, the PLE layer as ONE composition — the LAST block seam | no | [#2336](https://github.com/mudler/vllm.cpp/issues/2336) |
| W5f | `Qwen4ExpTextModel::Forward` — THE LAYER LOOP, and the `lm_head` tail | **yes** — `ModelRegistry::Forward` calls it on a loaded `qwen4exp` GGUF | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), [#2336](https://github.com/mudler/vllm.cpp/issues/2336) |
| W5g | the STATED n-gram vocabulary as the layout's authority, and a heap over-read closed on the GGUF arm | **yes** — `Qwen4ExpPleLayout`, reached from `qwen4_exp_forward.cpp` inside the loop W5f wired | [#2031](https://github.com/mudler/vllm.cpp/issues/2031) |
| W5h | the indexer side cache sized at ONE ROW PER TOKEN: `compress_ratio` 1, not 4 | **yes** — `make_kv_cache`, the same production hook W5c-1 reaches | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W5h's own issue OWED |
| W5i | the indexer side cache PAGED: the engine's fused MLA page + group 2's own block table, gathered and scattered with `vt::IndexSelect`/`vt::IndexCopy` | **yes, and W5j closed the gap this row named** — `ModelRegistry::Forward` reached the translation over a per-call scratch (M4a, M4b); since W5j a by-name step reaches it over the engine's OWN group-2 buffer through group 2's OWN gathered table | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), [#2249](https://github.com/mudler/vllm.cpp/issues/2249), W5i's own issue OWED |
| W5j | the forward CONSUMES the by-name channel, and `ModelFactory::consumes_multi_kv` narrows the engine's `multi_kv` refusal to a model-declared capability | **yes** — `ModelRegistry::Forward` dispatches a THREE-GROUP topology to this hook, which resolves all five published caches by name and reads group 2's own block table; M4 proves the guard still refuses with the bit cleared, M6 proves the tower is entered. It is still a SINGLE-SHOT prefill: nothing decodes a second token | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), [#2353](https://github.com/mudler/vllm.cpp/issues/2353), W5j's own issue OWED |
| W5k | the PLE conv ring's DTYPE and the n-gram history's RESIDENCY settled against the running lane oracle, and the SECOND STEP | **yes, and it DECODES** — `ModelRegistry::Forward` runs a prefill at `past_len` 0 and then a decode at `past_len` 6 over the engine's own persistent recurrent group, sampling a token on each; M1 deletes the n-gram write-back INSIDE `RunQwen4ExpPleBlock` and the step-1 history assertion reds, which is the first measured reach of that block's body | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W5k's own issue OWED |
| W5L | `GPUModelRunner` and `LoadedEngine` DRIVEN end to end, and the model-declared concurrency ceiling that keeps a server alive | **yes, and it SERVES** — a real `GPUModelRunner` allocates all three published groups, gathers all three block tables and runs a prefill then a decode through `execute_model` / `sample_tokens`; `LoadedEngine::FromModelDir` loads a `qwen4exp` GGUF and `generate` returns tokens; `examples/server` answers `POST /v1/completions` on CPU. M1 deletes the runner's `multi_kv` handoff, M3 deletes the per-group gather call site, and M4 deletes the clamp's production call site — each reds | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W5L's own issue OWED |
| W5n | the RELEASED `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact driven through `examples/server` on `thor:gpu0` — the first published `qwen4exp` bytes this row has ever read | **LOAD yes, TOKEN no** — all three shards load on `--device cpu` in 4446 s at 69.206 GiB peak RSS with every encoding keeping its blocks, the engine sizes its caches and the server answers `/health`; the first forward then refuses the artifact by name (`qwen4_exp_gated_residual: input_mix_weight_down must be float`, `src/vt/ops.cpp:2552`) because the file stores 194 hyper-connection mix weights as Q8_0, and `/v1/completions` returns 500. **Zero tokens.** No code changed; the defect is recorded, not worked around | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W5n's own issue OWED |
| W5p | the hyper-connection mixer takes a QUANTIZED mix weight: `mix_down`, `mix_up` and `block_inject` may keep the file's blocks and route through `vt::MatmulBT`, while `hc_norm_w` and the stream stay float and a block-typed one is refused by name | **yes** — `ModelRegistry::Forward` runs a prefill AND a second prompt over a `FixtureOpts::hc_mix_q8_0` file whose `hc_*_down`, `hc_*_inject` and `output_hc_down` are Q8_0, sampling a token that is the row's own maximum and logits that MOVE on the second prompt. M1 restores the pre-wave refusal and that case reds with the exact string the RELEASED checkpoint threw, which is what makes the reach measured rather than assumed. **W5q ran the released checkpoint through this repaired path: the refusal is GONE and the output is DEGENERATE** — see the W5q row | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W5p's own issue OWED |
| W6-CUDA | the first CUDA arms of this architecture: `vt::Qwen4ExpPleConv`, `vt::Qwen4ExpPleGate`, `vt::Qwen4ExpGatedResidualWriteBack` | **no, and VACUOUSLY so** — `ModelRegistry::Forward` is all-or-nothing and four ops plus `vt::RmsNormGroup` plus the block-decoding n-gram gather still have no CUDA arm, so no `qwen4_exp` step can reach a CUDA queue at all. The gate RAN on TWO architectures. `thor:gpu0` (`sm_110`, nvcc 13.0.88): 12 cases, 323 assertions, 322 passing, with the CPU arms matched BITWISE at 0 of 8772, 0 of 20480 and 0 across all 30 dtype combinations; 5 of 6 mutations red and M5 a compiler proof. The single failure was this suite's OWN oracle bound, which was re-derived and bitwise-backstopped. `dgx:gpu0` (`sm_121a`, GB10, nvcc 13.0.88) then ran the corrected suite green: 12 cases, 351 assertions, 351 passing, every rc READ rather than derived, and `cuobjdump` reporting `cuda_qwen4_exp_ple.cu.1.sm_121a.cubin` so the objects are genuinely built for that architecture. The two runs' mutation counts agree. Full result in the W6-CUDA section of `## Owed` | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W6-CUDA's own issue OWED |
| W5-LOADIO | `VT_LOAD_STATS` reports on the GGUF branch, and `PrefaultBorrowedSpan` counts BYTES rather than only spans | **yes** — the timing and `ReportGgufLoadIo` sit on the `.gguf` branch of `LoadedEngine::FromModelDir`, the production loader entry point, so `VT_LOAD_STATS=1` on any GGUF model now prints `mmap+header`, `weights` and the prefault's bytes/seconds where it previously printed NOTHING. It deliberately does NOT call `ReportLoadBytes`, whose three counters are incremented on the safetensors path only and would print zeros for an artifact the load had just moved 67.56 GiB of. Gated by a new case in `test_gguf_keep_quant.cpp` asserting an EQUALITY over a double load, because `> 0` is satisfied by a counter wired to the span count, to a constant, or to the last span alone | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), its own issue OWED |
| W5q | the RELEASED `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact driven through `examples/server` on the COMPOSED W5p+LOAD-IO tree, staged to worker-local disk | **SERVES yes, USABLE TOKEN no** — the W5n refusal is gone: a 5-token prefill and eight decode steps run with `model_executed=1` each, nothing throws, and `POST /v1/completions` returns **HTTP 200** with 8 completion tokens where W5n got a 500. **Every one of those tokens is id 0**, which this checkpoint's own `tokenizer.ggml.tokens` gives as `!`, and the answer is BYTE-IDENTICAL for two prompts of different lengths. So the forward is degenerate and prompt-independent on the real weights. Load 61 s from local disk (against 4446 s from CIFS), `VmHWM` 73.935 GiB, gate 72 cases / 10,380 assertions / 0 failed / 0 skipped with the oracle golden unmoved at `0.00982457`. **The CAUSE is NOT identified**: `logprobs` was not requested, so nothing distinguishes NaN, zero and constant logits, and no per-op probe was run | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W5q's own issue OWED |
| DECODEDIV | the forward decodes on `ModelForwardInput::device_token_ids` instead of the host `token_ids` the asynchronous runner leaves stale for decode rows | **yes, and it FIXES the CUDA arm's output** — `ForwardQwen4ExpForConditionalGeneration` is the production hook, and on `thor:gpu0` over the released UD-IQ1_S artifact through `examples/server` with NO `CUDA_LAUNCH_BLOCKING` the `--device cuda` tokens go from `11751 271 271 271 271 271 0 0` to `11751 13 15767 411 1928 11 628 567`, against a CPU control of `11751 13 15767 411 2029 11 1092 369` re-taken on the same tree. The gate runs on a CPU QUEUE and still convicts, because the defect is which array the hook reads; the fix made inert reds it and a byte-for-byte restore greens it. **NOT token-exact** — the residual is measurable in the PREFILL hidden state at about 0.3% and is [#2547](https://github.com/mudler/vllm.cpp/issues/2547), not this | [#2496](https://github.com/mudler/vllm.cpp/issues/2496) |
| W5r | the shared `dense_attn::ResidentWeight` stops dropping the load-time repack markers (`repacked`, `elem_kn_repacked`), and refuses to stage an `elem_kn_repacked` weight to a device | **yes, and W8CONFIRM PROVED IT IS THE FIX** (W5s asserted it; its `701606e51`-vs-`52f7ccbfc` comparison spans W5p too and cannot apportion) — `dense_attn_block.h:235-236` sits on the path `qwen4_exp_forward.cpp` takes for every hyper-connection mix weight, so on an aarch64 i8mm host the mixer stops reading `block_q8_0x4` bytes as flat `q8_0` across 48 layers x 2 sides plus the terminal mixer. W5r itself could neither run nor gate this: it was CPU-only on x86, where `vt::cpu::QuantRepackActive()` is false (`cpu_quant_repack_arm.cpp:275`) and the whole chain is inert, so its gate sets the flag BY HAND and asserts propagation. **W5s ran it on `thor`, where the chain is live** | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W5r's own issue OWED |
| W5s | the RELEASED UD-IQ1_S artifact re-driven on `origin/main` `52f7ccbfc` (W5p **and** W5r), four arms, one build, one staged copy | **SERVES yes, and the TOKENS ARE REAL** — `" Paris. Given this fact, what is"` and `" 100°C at sea level"` for two different prompts, eight distinct ids none of them 0, against W5q's eight consecutive id 0 on the same box and artifact. Reusing W7DIAG's read-only probe, the pre-W5r tree and this one agree numerically on `embed` and `after_widen` and diverge at exactly `stream.after_layer_0` (`nan=51200` -> `nan=0`); `LOGITS` was `zero=248320` with no maximum and is now `min -9.89818 max 15.7873`, argmax id 11751 = the `" Paris"` token. `VT_CPU_QUANT_REPACK=0` is byte-identical to the default, which is the correct outcome for a performance transform and the thing that was false before W5r. **NOT a token gate** (no oracle decoded these prompts; llama.cpp aborts in `build_delta_net_chunking`), no speed number, UD-IQ1_S only, one sequence. Lands NO product code | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W5s's own issue OWED |
| W8CONFIRM | the SAME artifact on the SAME `52f7ccbfc` tarball, TWO binaries differing only in `dense_attn_block.h:235-236`, four arms in one lease | **CAUSE ISOLATED** — `X-ON` (fix reverted, repack ON) returns `"!!!!!!!!!!!!!!!!"` with `LOGITS zero=248320` and `after_layer_0 nan=51200`, while `X-OFF` (same binary, repack OFF) and both `M` arms return `" Paris. Given this fact, what is the capital of France?\n\n<think>\n"` bit-identically. Same binary either side of one environment variable, so the defect needs the repack chain ACTIVE and the markers DROPPED; W5p is in all four arms and cannot explain a difference between them. Four prompts across factual, narrative and code, all correct, one ending on the model's own EOS. Binaries `e18a38a6…` vs `cfdf47bd…`, mutation applied-proof `2 -> 0` fix lines. **NOT a token gate**, n=1, UD-IQ1_S only, `--device cpu` only. Lands NO product code | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), W8CONFIRM's own issue OWED |
| W6-CUDA-B | the FOUR remaining CUDA arms: `vt::Qwen4ExpGatedResidual` (the mixer), `vt::RmsNormGroup`, `vt::Qwen4ExpQsaCompress`, `vt::Qwen4ExpQsaGatherAttention` | **no, and VACUOUSLY so** — `ModelRegistry::Forward` is all-or-nothing and `EmbeddingKernelCuda` still refuses a block-quantized table by name, so no `qwen4_exp` step can reach a CUDA queue. What CHANGES is the size of the gap: the three W6-CUDA arms sit immediately AFTER these four in the same functions (`qwen4_exp_ple_block.cpp:489,499` before `:531,580`; `qwen4_exp_forward.cpp:418,476` before `:457,506`), so seven kernels go from four separate blockers to ONE. The wave was dispatched as THREE ops; `kQwen4ExpGatedResidual` was missed because `kQwen4ExpGatedResidualWriteBack` contains it as a substring, and two independent audits caught it before any code was written. **AND THE VACUITY IS MEASURED, NOT ASSERTED**: mutation M8 deletes the production call site and `test_qwen4_exp_ple_block` reds (rc 1, so the site is LIVE) while both CUDA suites stay green (rc 0, so nothing reaches the new arms). Gated on `thor:gpu0` `sm_110`, nvcc 13.0.88, on the MERGE RESULT `12071a5dd`: red 5/7, 10/10 and 4 assertions failing for the intended reason, green 7/70, 10/120 and 12/351 all `SUCCESS!`, eight mutations applied and built and seven RED, `RESTORE CHECK RC=0`, and `cuobjdump` reporting 2 of 2 `sm_110` cubins. The norm and the compressor are BYTE-IDENTICAL to their CPU arms (0 bytes differ across all 18 dtype triples; relL2 exactly 0 vs the golden); the gather's only divergence is `expf` at 2.384e-07 against a 2.560e-07 one-ulp bound. Full result in the W6-CUDA-B section of `## Owed` | [#2031](https://github.com/mudler/vllm.cpp/issues/2031), [#2380](https://github.com/mudler/vllm.cpp/issues/2380) |

Every `no` in that column has a named `## Owed` entry under AGENTS.md "Nothing
lands dead", and the qualified `yes` rows say what they reach rather than
claiming a decode. **W5k IS THE ROW THAT DECODES, and the sentence that stood
here is now false.** It read "Nothing in the table decodes a token, and W5f does
not change that", which was true of every wave up to W5j and stopped being true
the moment `past_len > 0` returned logits. A second step now runs through
`ModelRegistry::Forward` over caches that persist between calls, and samples a
token. **W5L IS THE ROW THAT SERVES, and this sentence has moved again.** It
read "What is still NOT claimed is SERVING: `num_reqs > 1` is refused, the
positional arm is one-shot, and `GPUModelRunner` has not been driven end to
end". The third is gone: a real `GPUModelRunner` allocates the three groups,
gathers all three block tables, publishes the five-name index and runs
`execute_model` / `sample_tokens` for a prefill and then a decode, and
`LoadedEngine::FromModelDir` plus `examples/server` answer a
`POST /v1/completions` on CPU. The first two remain, and the FIRST of them has
changed CONSEQUENCE rather than status: `num_reqs > 1` is still refused, and
because an EngineCore that meets that refusal DIES rather than degrades, the
factory now declares `serves_one_sequence_per_step` and the engine clamps
`--max-num-seqs` to 1. What is still not claimed is a BATCH, a real checkpoint,
and any number at all. W5f's own contribution is unchanged by this:
it changed the reason the rows above were unreached. Before it, most of the rows above were
unreached because nothing composed them; after it, the composition exists and
runs from a production entry point, and what is missing is the SECOND STEP. The
count is deliberately not written out: this section deletes prose counts of its
own table, and W5g, W5h and W5i each added a row without touching this sentence,
which is exactly the drift the policy exists to stop. The policy holds for the
TABLE and it did not save the PROSE below: W5i found a stale enumeration further
down that survived W5g, and repaired it in flow.
**WHAT W5f's LOOP ACTUALLY REACHES IS A PREFIX OF THAT COLUMN, NOT ALL OF IT,
and the sentence that stood here said all of it.** It read "Every `no` above is
now reached THROUGH W5f's loop at a single-shot prefill". That is false, and the
mutation record's own prose is the more careful one: the reachability case
"reaches layer 1's PLE block". The shared GGUF fixture is internally
inconsistent (`## Owed`), so the production path stops at
`Qwen4ExpPleLayout`'s vocabulary-size cross-check
(`qwen4_exp_ple_block.cpp:129`), which is called at
`qwen4_exp_forward.cpp:391` — one line BEFORE `RunQwen4ExpPleBlock`. The fixture
puts its only PLE layer at index 1 and PLE runs FIRST in a decoder layer
(`:1218`), so the prefix that runs is exact:

**THIS ENUMERATION WAS STALE AND W5i's MUTATION BATTERY IS WHAT CAUGHT IT.** It
read that the loop stopped at layer 1's `Qwen4ExpPleLayout` refusal and that "the
whole Qwen Sparse Attention arm" was reached by nothing, "because the fixture's
QSA layer is layer 3 and the loop never gets there". W5g removed the refusal that
stopped it — it made the STATED n-gram vocabulary the layout's authority — and
nothing rewrote this list afterwards. Two measurements on `e12a197cd`'s tree say
so directly, and both are in W5i's mutation record above:

- `test_qwen4_exp_layer_loop.cpp:751` MESSAGES `qwen4_exp sampled token id: 15 of
  16 (logit range [3290.84, 95090.7])`. `ModelRegistry::Forward` RETURNS LOGITS
  on this fixture; it does not refuse partway.
- W5i's M4a made `IndexerRows` — inside `RunQwen4ExpQsaBlockPaged`'s indexer —
  fatal, and `REQUIRE_NOTHROW( fl = vllm::ModelRegistry::Forward(*model, in) )`
  THREW. The QSA arm is therefore reached FROM A PRODUCTION ENTRY POINT.

`RunQwen4ExpPleBlock`'s body follows by construction rather than by its own
mutation: PLE runs FIRST in a decoder layer (`:1218`), the fixture puts its only
PLE layer at index 1, and the loop demonstrably reaches layer 3's QSA, so layer
1's PLE body ran. That inference is recorded as an inference.

**W5k REPLACED THAT INFERENCE WITH A MEASUREMENT, and the sentence below it is
now out of date.** It read "No mutation on this row has deleted
`RunQwen4ExpPleBlock`'s call site", so the PLE ops were reached-by-argument. W5k's
M1 deletes the n-gram history WRITE-BACK inside `RunQwen4ExpPleBlock`'s own body
— `update_conv_state(..., state_idx=2)`, `modeling_qwen4_exp.py:1089-1091` — and
`test_qwen4_exp_layer_loop.cpp:1874` reds with `CHECK( 0 == 7 )` on a step driven
through `ModelRegistry::Forward`. A body that did not run could not have failed to
write. W5k's M2 (the EOS seed, in the same body) additionally reds the oracle
golden at 0.777988 against a bound of 0.03. So the PLE block body is
reached-by-measurement from a production entry point, and the paragraph that
follows describes the state before this wave:

**(superseded)** No mutation on
this row has deleted `RunQwen4ExpPleBlock`'s call site, so the PLE conv and gate
ops (W5b-3, W5e-1, W5e-2) were reached-by-argument and not reached-by-measurement,
which is a weaker claim and is written as one.

- **Reached through `ModelRegistry::Forward` today:** the loop end to end on the
  fixture — layer 0 whole (the attention hyper-connection and its rank-1
  write-back, W5b-2/W5b-6; `RunGdnBlockPaged`, W5b-1; the MLP hyper-connection;
  `RunQwen4ExpMoeBlock` and its adapter, W5d-4), layer 1's PLE, and the Qwen
  Sparse Attention arm (W5b-4, W5b-5, W5d-3, and W5i's paged indexer side cache),
  through to the `lm_head` and a sampled token.
- **Reached by NOTHING in production, at this merge commit:** the terminal
  `use_combine=false` hyper-connection mixer at `:1430`, which is after the loop.
  **THAT IS THE WHOLE LIST NOW.** This bullet carried a second item — the
  ENGINE's group-2 buffer, unreached "because `ModelRegistry::Forward` refuses
  `multi_kv` and the registry hook substitutes a per-call scratch" — and W5j
  removed it: a step carrying `multi_kv` reaches the hook, which hands the block
  group 2's own buffer through group 2's own gathered table. The permutation is
  therefore exercised by the allocator's pages and not only by the block's gate,
  and `test_qwen4_exp_layer_loop`'s W5j case asserts the written ROW SET rather
  than a value, because a value gate cannot see a paging defect.

**THE `Reached?` COLUMN IS THE AUTHORITY AND THIS PROSE IS NOT A SECOND COUNT.**
Rows that still read `no` do so for the ordinary reason — nothing composes them
— and not because the loop stops. What remains true from the sentence this
replaces is the SCOPE of the claim, and W5k then W5L each narrowed what that
scope EXCLUDES. It read "this is ONE single-shot prefill of ONE sequence, it is
not a decode": since W5k it is a prefill AND a decode, and since W5L both are
driven by the engine rather than assembled by hand. ONE SEQUENCE is the part that
has not moved, and rewriting any `no` to `yes` on the strength of a served
request would be the overstatement this section exists to prevent — a served
request proves the composition RUNS, never that an unreached op is reached.

**Reached, and LOADING — on a CPU device:** a `qwen4exp` file lands on
`Qwen4ExpHfConfigFromGguf` through the `kGgufArchArms` dispatch row, the registry
resolves `Qwen4ExpForConditionalGeneration`, and W5a's `load_weights` hook now
materializes the whole text tower instead of refusing. That is the first
production call site this row has had. On any device that cannot gather from a
block table the load REFUSES BY NAME ahead of any tensor I/O, because the
n-gram table would otherwise expand from 26.822 GiB to 95.368 GiB of host
memory ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)); the CUDA
gather arm is owed.

**Reached, and no longer refusing: the KV-cache spec.** W5c-1
([#2031](https://github.com/mudler/vllm.cpp/issues/2031)) makes
`make_kv_cache` return three groups — the QSA layers' paged K+V, ONE uniform
recurrent group carrying `[gdn_conv, temporal, ple_conv, ngram]` on every
linear layer, and the QSA indexer side cache as an `MLAAttentionSpec` at
ONE ROW PER TOKEN since W5h — over real per-layer names, so the runner takes its
multi-cache path and allocates every published cache. The engine half was
`ENG-RECURRENT-MULTISTATE` (#2131); the second half that row expected to be
needed, more than one recurrent group, is NOT on this path, because upstream
declares one recurrent shape model-wide. **THE THREE THINGS THIS PARAGRAPH SAID IT DOES NOT DO ARE NOW TWO
DONE AND ONE STANDING, AND W5L REPAIRS IT IN FLOW.** It read: "the n-gram history
the runner allocates is ZERO-SEEDED where it needs EOS and nothing on that path
corrects it — W5e-2 seeds it inside `RunQwen4ExpPleBlock`, and no forward calls
that block, so the correction does not yet reach the runner's own row; nothing
READS the side cache's block table yet (W5c-2 made the runner gather it and hand
it to the forward; no consumer takes it); and every byte figure is derived on a
CPU host rather than measured on a device." W5f gave the block a caller, W5j gave
the side cache's table a consumer, and W5L measures both on the RUNNER's own
buffers: `test_qwen4_exp_runner.cpp` reads the runner-allocated n-gram history
after a prefill and finds the prompt's last two ids in it rather than zeros, and
deleting `gather_group_block_tables`'s call site reds that same case by name.
What SURVIVES is the third: every byte figure here is still derived on a CPU host
and none is measured on a device. W6-CUDA has since given `vt::Qwen4ExpPleConv`,
`vt::Qwen4ExpPleGate` and `vt::Qwen4ExpGatedResidualWriteBack` their first CUDA
arms, so "no CUDA kernel exists" is no longer the reason; the reason is that four
further `qwen4_exp` ops plus `vt::RmsNormGroup` and the block-decoding n-gram
gather still have none, and `ModelRegistry::Forward` is all-or-nothing, so no
`qwen4_exp` step reaches a CUDA queue.

**"NO GATE HERE CAN SEE THAT" WAS TRUE WHEN IT WAS WRITTEN AND IS NOT TRUE NOW**,
so it is corrected rather than carried. W5e-2's mutation M2 replaces the EOS seed
with zero and reds 4 of 11 cases against the lane-pinned end-to-end golden, at a
measured separation of 1.2892 versus a 1e-5 tolerance. What survives is the
narrower statement above: the SEEDING is gated. **THE ROUTING IS GATED TOO SINCE
W5L, and the clause that said otherwise is corrected here.** It read "the ROUTING
of the runner's slot into it is not [gated], because there is no loop to route
it": there is a loop, the runner routes its own slot into it, and W5L's M2 sets
`caches.ple[i].state_row` one row wrong and reds four assertions — the history at
the runner's assigned slot stays all zeros while the prompt's tail is `{7, 2}`.

**(superseded by W5k and W5L)** This paragraph read "**Reached, and still
refusing:** the forward. Nothing decodes a token, so there is still no token
number, no speed number, no `examples/server` e2e and no `docs/USAGE.md` weights
row". Three of those four are now wrong. The forward decodes (W5k), and W5L
serves a `POST /v1/completions` through `examples/server` on CPU over a synthetic
`qwen4exp` GGUF. **THE `docs/USAGE.md` WEIGHTS ROW IS PAID, BY W5n, AND WHAT IT
RECORDS IS A REFUSAL.** This paragraph read "no byte of
`unsloth/Qwen3.8-Flash-Next-GGUF` has been served on any host this row reaches",
and that stopped being true on 2026-08-30. `rc` job `0f188dd1` on `thor:gpu0`
loaded the released UD-IQ1_S artifact — 67.564 GiB, 3 shards, 1224 tensors —
through `examples/server` on `--device cpu`, and the server listened. **The
LOAD is the good half and it is real**: 4446 s to `/health`, peak RSS `VmHWM`
69.206 GiB against a 67.564 GiB file, with every one of the file's nine
encodings keeping its blocks (anonymous memory moved 4 → 11 GiB across a load
whose n-gram table alone would have added 95.368 GiB). **That 4446 s is a
FILESYSTEM number and the sentence that quoted it alone was misleading**:
LOAD-IO measured the same artifact at 60 s from worker-local disk on the same
box, and W5q reproduced 61 s independently, so our loader's own host work is
~1.3% of it and the rest is the CIFS mount. **The FORWARD then refused the
artifact by name and ZERO tokens came out**, and W5p removed that refusal at its
source. **W5q RE-RAN THE ARTIFACT AND THE REFUSAL IS GONE**, so this paragraph's
verdict has moved rather than been deleted: `POST /v1/completions` now returns
200 with eight tokens instead of a 500, and every one of those tokens is id 0
with a BYTE-IDENTICAL answer for two different prompts. There is still NO token
number and NO speed number, and the reason has moved twice — from "nothing has
been read", to a named refusal, to a degenerate forward on the real weights
whose cause nothing has yet identified. See `## Owed`. W2, W3 and W4 remain host
reference math with no production call site.

**W5b-6 ([#2218](https://github.com/mudler/vllm.cpp/issues/2218)) closes the
gamma polarity and it does NOT decode.** `vt::Qwen4ExpGatedResidual` now takes
the RAW HuggingFace gamma and adds the 1 itself, which is the convention the
other three consumers of this architecture's gammas already had, so the layer
loop can hand it `Qwen4ExpWeights` directly instead of scaling the
hyper-connection stream by ~0. The gate is
`tests/vllm/models/test_qwen4_exp_forward.cpp`, the first **`qwen4_exp`** suite
that LOADS a gamma through `ModelRegistry::Load` and runs it through a device op
in one case — which is why eleven single-sided waves of THIS row could not see
it. **THE UNSCOPED FORM OF THAT SENTENCE WAS FALSE AND IS CORRECTED HERE.** It
claimed the first such suite in the tree; it is not.
`tests/vllm/models/test_nemotron_h_paged_forward.cpp` and
`tests/vllm/models/test_kimi_linear_paged.cpp` each call `ModelRegistry::Load`
inside a `TEST_CASE` and drive the loaded weights, gammas included, through the
device ops of a forward. The claim that survives is the narrow one, and it is
the one the argument needed: no `qwen4_exp` suite had ever composed the loader
with an op, so the contradiction between them was unreachable here. The synthetic
`qwen4exp` file moved to `tests/support/qwen4_exp_gguf_fixture.h` so the loader
suite and the forward suite share ONE builder.

**AND IT CORRECTS THIS SECTION'S OWN CLAIM.** The paragraph below used to say
"THE OP AND SEAM WORK IS FINISHED; WHAT IS LEFT IS THE LAYER LOOP." That is not
true. Five things the loop composes were absent from `main` when W5b-6 surveyed
it — a standalone grouped RMS norm for PLE's three norms, a PAGED QSA consumer,
the group-2 block table (W5c-2, landed: the runner gathers every published
group's table), a MoE weight adapter, and an externally linked mRoPE builder — and `ModelRegistry::Forward` additionally refuses every
multi-cache topology by name, which is what this model publishes. Each is
measured and cited under `## Owed`, and the production refusal in
`qwen4_exp_registry.cpp` now names them instead of naming W2, W3 and W4, which
landed. **A wave dispatched to "write the layer loop" will not decode a token;
it has these prerequisites, at least two of which (the grouped norm, the paged
QSA arm) are op-sized waves of their own.**

**ALL FIVE ARE NOW CLOSED, AND THE COUNT IS STATED HERE RATHER THAN LEFT TO A
READER TO RECOUNT.** The stale enumeration is
[#2288](https://github.com/mudler/vllm.cpp/issues/2288), filed for traceability
and FIXED IN THE SAME FLOW by
[#2265](https://github.com/mudler/vllm.cpp/pull/2265), the wave this correction
first rode with.

- **Item 1, the grouped RMS norm**, is `vt::RmsNormGroup`, landed by W5d-1
  ([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 1) as
  `25ee19464`.
- **Item 5, the externally linked mRoPE builder**, is `BuildMropeCosSinHost`,
  landed by W5d-2 (#2249 item 5) as `3ed2378a3`; that wave corrected the
  paragraph above and did NOT correct this list or the production refusal
  string, so both had been naming a finished seam since it merged.
- **Item 4, the MoE weight adapter**, is
  `src/vllm/model_executor/models/qwen4_exp_moe.{h,cpp}`, landed by W5d-4 (#2249
  item 4) as `3f9177f7f`. Closed as a SEAM, not as a call: W5d-4 landed
  unreached and says so under `## Owed`, exactly as W5d-1 does.
- **Item 2, the PAGED QSA consumer**, is `Qwen4ExpQsaPagedCaches` +
  `RunQwen4ExpQsaBlockPaged` over a `kv_block_table`/`kv_block_size` ADDRESS
  MODE inside `vt::Qwen4ExpQsaGatherAttention`, landed by W5d-3 (#2249 item 2) —
  the wave this section is being merged with, which is why this recount rides
  here. Closed as a SEAM as well, and at the time only for the K/V half: the
  INDEXER side cache was still contiguous and its paged STORE outlived the
  survey. **W5i closed that half too** — the side cache is the engine's fused MLA
  page now, stored and read through group 2's block table — so what outlives the
  survey is narrower again: the engine's group-2 BUFFER still does not reach the
  block, which is W5j's and is named below.
- **Item 3, the group-2 block table**, is
  `GPUModelRunner::gather_group_block_tables`, landed by W5c-2 (#2249 item 3) —
  the wave THIS section is being merged with, which is why this recount rides
  here a second time. Every published group's table is gathered on the
  multi-cache path and published by group id on `MultiKvCacheIndex`. Closed as
  a SEAM, like the other four: the MAP reaches the forward and no consumer
  reads it, which is the narrower entry `## Owed` now carries in its place.

**NONE remain. The count is ZERO.** What refuses is no longer a prerequisite:
it is the LAYER LOOP itself, `Qwen4ExpTextModel::Forward`, owned by
[#2031](https://github.com/mudler/vllm.cpp/issues/2031), which nothing above
substitutes for. Two things sit beside it and neither is one of the five. The
QSA indexer side cache's PAGED STORE **is no longer one of them: W5i landed it**
— the map arrived with W5c-2, and `Qwen4ExpQsaPagedCaches::index_key` is the
engine's own `[num_pages, block_size, indexer_head_dim]` page addressed through
group 2's table. What sits beside the loop instead is that the engine's group-2
buffer does not REACH the block, because `ModelRegistry::Forward` refuses
`multi_kv`; the hook allocates a scratch in that same paged shape meanwhile. And
the `multi_kv` refusal is
not this row's: `ModelRegistry::Forward` refuses every multi-cache topology by
name and this model publishes one, which #2249 records as belonging to an
engine row. The refusal in `qwen4_exp_registry.cpp` says exactly this at this
merge commit, and the emitted bytes were read back out of the running hook to
prove it.

**SUPERSEDED BY W5j, AND THE PART THAT WAS WRONG IS THE OWNERSHIP.** The
paragraph above is kept because it is the argument that scoped two waves, but two
of its statements no longer describe the tree. The engine's group-2 buffer DOES
reach the block, over group 2's own gathered table. And "the `multi_kv` refusal
is not this row's" was true of the GUARD and false of the FIX: #2353 resolved it
as a per-architecture capability each model row owns an arm of, so
`ModelFactory::consumes_multi_kv` and its arm landed HERE, in the row that ports
this architecture. The guard itself is still the engine's and still refuses
DeepSeek-V4 and GLM-5-Next. The refusal bytes were read back out of the running
hook again for W5j — see the mutation record's M1 and M4, which quote them.

**AND THIS PARAGRAPH CONTRADICTED ONE ELEVEN LINES BELOW IT, WHICH IS #2288 IN
ITS SEVENTH TURN AND IN A SHAPE THIS ROW HAD NOT PRODUCED BEFORE.** Not a stale
enumeration, but TWO LIVE ENUMERATIONS THAT DISAGREE: "NONE remain. The count is
ZERO" above, and "What has no production shape yet is the PLE block, the GDN
weight adapter onto `GdnLayerWeights`, the hyper-connection stream through the
per-layer loop, and the loop itself" below. Both were on `c0fa299b1` and a reader
took whichever they reached first. A wave dispatched to write the layer loop read
the first one, measured the tree instead, and returned `NEEDS_DECISION` rather
than the loop. The measured reconciliation is
[#2336](https://github.com/mudler/vllm.cpp/issues/2336), and it moves the count
in BOTH directions.

**Both sentences are true of different things, and the sentence that was missing
is what makes them consistent.** #2249 surveyed FIVE PREREQUISITES, not the whole
gap. All five are closed and the count of five is zero — that part stands. It was
never a statement that nothing else was missing, and it read as one. Three
measured corrections, each on `bd90b92b0`:

- **The PLE GATE was op-sized and NOTHING had ever named it.**
  `git grep -n 'clamp_min\|signed_sqrt\|SignedSqrt\|copysign' src/vt include/vt`
  returned ZERO lines, so `modeling_qwen4_exp.py:1181-1182` — the signed square
  root and the sigmoid that scales `value` by it — had no `vt::` expression at
  all; the only implementation was the host `float`→`float` `SignedSqrtGate`
  (`qwen4_exp_ple.cpp::SignedSqrtGate`), whose single caller is `PleForward` in
  the same translation unit. W5e-1 lands it as `vt::Qwen4ExpPleGate`. It was
  never one of the five, so closing all five could not have supplied it, and the
  production refusal's "the ops and block seams ARE on main" was therefore an
  OVERSTATED refusal — #2254's polarity, not #2276's. Repaired in the same flow,
  with the emitted bytes read back out of the running hook.
- **The DOT and the flatten around it need NO new op, and saying so is half the
  point**, because the wave that writes this must not add general ops it does not
  need. The per-`(t, j)` dot at `:1180` is `vt::BatchedMatmul` over `[T*hc, 1, H]
  x [T*hc, H, 1]` VIEWS of the two `[T, hc*H]` buffers — only the innermost dim
  must be unit-stride — and `test_qwen4_exp_ple_gate.cpp` RUNS that composition
  against the golden rather than asserting it. Two ops were checked and neither
  can serve the multiply, because BOTH of its operands broadcast:
  `vt::SigmoidGateBf16` refuses by count ("sigmoid_gate_bf16: out/attn/gate must
  have the same element count") and `vt::MulColVecF32` scales per output COLUMN
  where this scales per row. So ONE fused op was owed, not five general ones.
- **Two items the sentence below lists as missing production shapes are smaller
  than that phrase implies** (#2336 §3, §4). The GDN weight adapter onto
  `GdnLayerWeights` is a FIELD COPY — nine assignments and one rename, because
  the qwen4_exp and qwen3_5 GGUF loaders read the same tensor names and land on
  the same orientation with `gdn_expand_nk` on — with three non-arithmetic risks
  (`output_gate_type` is sigmoid here and silu there and `ParseQwen4ExpParams`
  DISCARDS it; a per-step adapter copy loses `ResidentWeight::d_dev` and
  re-uploads the tower; `in_proj_ba` stays empty so `vt::GdnPackedDecode` never
  fires, which is exact parity with qwen3_5's GGUF path and a perf ceiling, not
  a defect). And the hyper-connection widen `hidden_states.repeat(1, 1, hc_count)`
  (`:1412`) is `vt::IndexSelect` with `idx = [0,0,0,0,1,1,1,1,...]`, so it is
  loop work rather than seam work.

**WHAT IS ACTUALLY LEFT, AFTER W5e-2: the LOOP, and only the loop.** The
sentence this replaces was written on W5e-1's branch and read "WHAT IS ACTUALLY
LEFT, AFTER W5e-1: the PLE BLOCK, and then the loop", naming
`RunQwen4ExpPleBlock` as the missing symbol so that it would resolve the day
W5e-2 landed. It has. **The count of missing BLOCK SEAMS is ZERO and each one
resolves to a symbol**, read off this tree rather than off any wave's prose:
`RunQwen4ExpQsaBlock` / `RunQwen4ExpQsaBlockPaged` (W5b-5, W5d-3),
`RunQwen4ExpMoeBlock` (W5d-4), `RunQwen4ExpPleBlock` (W5e-2,
`qwen4_exp_ple_block.{h,cpp}`).

**W5e-2 IS WHAT MAKES THREE PREVIOUSLY CALLERLESS SEAMS REACHABLE FROM A BLOCK,
and the distinction between that and REACHED is the whole of this row's honesty
about itself.** Before it: `PleForward` had zero callers outside its own
translation unit, `vt::RmsNormGroup` had zero callers outside `src/vt/` and its
own suite even though W5d-1 landed it FOR PLE's three grouped norms,
`vt::Qwen4ExpPleGate` had none at all, and the n-gram gather had no composition
under `src/` (`BuildNGramIds` is host and `PleForward` was its only caller).
After it, `vt::RmsNormGroup` and `vt::Qwen4ExpPleGate` go from zero callers
under `src/` to one, and `BuildNGramIds` gains its first caller OUTSIDE ITS OWN
TRANSLATION UNIT — it has two under `src/`, `PleForward`
(`qwen4_exp_ple.cpp:381`) and this block (`qwen4_exp_ple_block.cpp:276`), which
is what the four lines above already say. In all three cases the new caller is
this block. **The block itself still has none**, so nothing here is reached from a
production entry point and the reachability mutation for this wave is VACUOUS
rather than passing; the row says so in its own mutation record and the
production refusal says so in its emitted bytes.

**The two obligations `## Owed` recorded for this block are discharged
DIFFERENTLY, and both entries above now say which.** The `conv_mask` PAIRED
obligation is CLOSED — the block refuses a masked position whose id is not EOS,
by name, which is the first enforcer that half has had. The EOS seeding is
PERFORMED and one hop short: the block seeds on `past_len == 0`, mutation M2
proves the golden sees a zero seed, and nothing routes a runner's recurrent slot
into the block yet.

The loop is **W5f**, under
[#2031](https://github.com/mudler/vllm.cpp/issues/2031) and
[#2336](https://github.com/mudler/vllm.cpp/issues/2336). **IT HAS LANDED, AND
EVERY SENTENCE ABOVE THAT SAYS OTHERWISE IS SUPERSEDED BY THIS ONE RATHER THAN
DELETED**, because the argument they make — that the block seams being finished
is not the loop being written — is what W5f had to satisfy and is worth keeping
legible.

**WHAT W5f LANDS.** `Qwen4ExpTextModel::Forward`
(`src/vllm/model_executor/models/qwen4_exp_forward.{h,cpp}`) composes the
48-layer stack: the `embed_tokens` gather, the `repeat(1, 1, hc_count)` widen as
`vt::IndexSelect` over a repeat index, then per layer the PLE block FIRST on the
hc-wide stream, the attention hyper-connection, the Gated DeltaNet or Qwen
Sparse Attention arm, the rank-1 write-back, the MLP hyper-connection, the MoE
block and its write-back — then the terminal `use_combine=false` mixer with NO
final RMSNorm after it. Two seams it needed and nothing had: `Qwen4ExpGdnHfConfig`,
the GDN arm's config projection, and `Qwen4ExpGdnBlockWeights`, the field copy
onto `GdnLayerWeights`. The `lm_head` tail lives in the registry hook, because
`Qwen4ExpTextModel` carries no head — that is `Qwen4ExpForCausalLM`.

**THE GDN ARM IS THE QWEN3.5 BLOCK, AND THAT IS A MEASUREMENT RATHER THAN AN
ASSUMPTION.** `Qwen4ExpTextGatedDeltaNet` and `Qwen3_5GatedDeltaNet` are
BYTE-IDENTICAL at the pin — the whole class, `__init__` and `forward` — except
for one constructor argument, `activation=config.output_gate_type or
config.hidden_act` against the default. Measured by diffing the two classes out
of the installed 5.16.0 package (`modeling_qwen4_exp.py:403-564` against
`modeling_qwen3_5.py:387-547`, class bodies, one hunk). So `RunGdnBlockPaged` is
this architecture's linear-attention layer and a second GDN implementation would
have been the parallel path AGENTS.md forbids. What the one difference costs is
[#489](https://github.com/mudler/vllm.cpp/issues/489)'s axis and it is now gated;
see the mutation record.

**THE ORACLE IS STANDING, WHICH THE LANE PIN SAID IT WAS NOT.**
`.agents/oracles/transformers.md` records `gateable = no` for this lane with the
reason "no published artifact fits any fleet device; blocked on memory, not
software". That reason is about the RELEASED CHECKPOINT and it is still true. It
is not a statement about the architecture: a TINY RANDOM CONFIG of
`Qwen4ExpTextModel` runs end to end on CPU in seconds. W5f stands one up —
transformers 5.16.0 imported and its `modeling_qwen4_exp.py` sha256 ASSERTED
against `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`
before anything is observed — and `scripts/gen-qwen4-exp-forward-goldens.py`
emits `tests/vllm/models/qwen4_exp_forward_goldens.inc` from it. **The lane's
`gateable` line is NOT edited here**, because that field is about the checkpoint
this row must eventually serve and promoting it on the strength of a tiny config
would be the overstatement its own text warns against. What is now false is the
weaker reading a reader could take from it — that nothing about this
architecture can be gated against a running oracle — and this paragraph is the
correction.

**WHAT IT DOES NOT DO, AND THE BOUNDARY IS THE ENGINE'S.** No token is decoded.
`ForwardQwen4ExpForConditionalGeneration` serves a SINGLE-SHOT PREFILL of ONE
sequence at `past_len == 0` and refuses anything else BY NAME, with the emitted
bytes read out of the running hook. The reason is exactly the one #2336 recorded
and #2353 confirmed: `ModelForwardInput` carries two POSITIONAL cache channels,
`attn_kv` and `gdn_state` (`model_registry.h:439-440`), and the QSA indexer side
cache and the PLE layer's conv ring and n-gram history are NEITHER, so at
`past_len == 0` this hook allocates them as per-call scratch and at any other
`past_len` there is nowhere they could have persisted. The channel that would
carry them is `multi_kv`, refused for every model by `ModelRegistry::Forward`
(`model_registry.cpp:461-478`) — and #2353 established that refusal must NOT be
lifted yet, because none of the three arriving architectures has a consuming
forward and the by-name channel cannot address recurrent (`MambaSpec`) members
at all. `num_reqs > 1` is refused for the same seam reason:
`RunQwen4ExpQsaBlockPaged`'s `block_table` is i32 `[1, max_pages]`.

**ONE MORE CORRECTION #2336 CARRIES, because it bounds what any loop wave can
gate.** `## Owed` says "a forward reached through `ModelRegistry::Forward` with a
hand-built POSITIONAL cache set is gateable today". `ModelForwardInput` carries
exactly two positional cache channels, `attn_kv` and `gdn_state`
(`model_registry.h:439-440`); the QSA indexer side cache is NEITHER, and the only
channel that could carry it is `multi_kv`, which `ModelRegistry::Forward` refuses
by name (`model_registry.cpp:462-478` — #2336's body cites `:440-478` and its own
comment corrects the range). So the hedge holds only where the side cache can be
a PER-CALL SCRATCH, i.e. a single-shot prefill at `past_len == 0`. No multi-step
decode of this architecture is reachable in this tree until
[#1925](https://github.com/mudler/vllm.cpp/issues/1925) lands, and
`RunQwen4ExpQsaBlockPaged` additionally takes `block_table` as i32
`[1, max_pages]`, so `num_reqs > 1` is out of reach for the same wave.

**AND THE COUNT IS ZERO BY SET DIFFERENCE, WHICH IS NOT WHAT EITHER SIDE OF
THIS MERGE SAID ON ITS OWN.** `main` carries W5d-3, whose text removed item 2
and still listed the group-2 block table that W5c-2 closes; this branch's text
removed item 3 and still listed the paged QSA consumer that W5d-3 closed. Both
sides therefore said ONE, both were exactly one item too long, and taking either
side whole would have landed a survey naming finished work — #2288 again, in its
sixth turn on this row. Deleting the enumeration instead would have been the
opposite error, because a survey that stops counting reads as an unfinished one.
The same trap has now caught this branch three times, against three baselines:
against #2265, whose edit renumbered five to THREE while still listing the paged
consumer; against W5d-4, which renumbered to FOUR from the other side; and here,
against W5d-3. The remaining set is the survey minus EVERY landed wave, never
the shorter of two lists — and when that difference is empty, the survey says so
in words rather than by falling silent.

ONE SHAPE FROM W5d-3'S OWN EDIT SURVIVES THE RECOUNT AND IS WORTH KEEPING.
Its five-to-four step MERGED two items rather than dropping one — a statement
about that edit's arithmetic, not about the count here, which is zero. What
W5d-3 discharges is the K/V half of the paged axis; what survived of it is the
indexer side cache, which was already its own item and whose block table W5c-2
now gathers. A reader who counts items without reading them will conclude a
prerequisite vanished when it was only folded into the neighbour it shares an
axis with — and will now conclude the port is done when what closed is its
prerequisites and not its loop.

**What is owed, in order. THIS PARAGRAPH'S OPENING CLAIM WAS WRONG AND IS
CORRECTED ABOVE: the op and seam work is NOT finished.** What follows is still
the right list of what W5b-1..5 landed; what it got wrong is the inference that
nothing else was missing. W5b needed five slices and four of them are ops or
seams:
`RunGdnBlockPaged` for the 36 linear layers (W5b-1), the two gated-residual ops
for the 10240-wide stream (W5b-2), `vt::Qwen4ExpPleConv` (W5b-3) and the two QSA
ops (W5b-4). W5b-5 turned the last of those into a decoder-layer BLOCK —
`RunQwen4ExpQsaBlock`, the first production composition of the QSA indexer — so
nothing the QSA indexer needs is missing from the `vt::` surface any more —
which the sentence this replaces overstated into a claim about the whole
architecture. **THE QUALIFIER THAT FOLLOWED IT IS NOW FALSE AND IS REMOVED
HERE**: it said "though the PLE block's grouped RMS norm still is", and
`vt::RmsNormGroup` landed with W5d-1 (#2249 item 1, `25ee19464`). What was
missing from the `vt::` surface at that moment, and what nothing in this section
named, was the PLE GATE — see the recount above and
[#2336](https://github.com/mudler/vllm.cpp/issues/2336). W5e-1 lands it.
What has no production shape yet is the PLE BLOCK and the LOOP ITSELF. **THIS
SENTENCE USED TO LIST FOUR ITEMS AND THE OTHER TWO WERE MEASURED SMALLER THAN
THE PHRASE IMPLIES** (#2336 §3, §4, and the recount at the head of this
section): the GDN weight adapter onto `GdnLayerWeights` is a nine-assignment
field copy with three non-arithmetic risks rather than a second W5d-4, and the
hyper-connection stream's widen is `vt::IndexSelect` over a repeat index, so it
is loop work. Neither is a seam. The PLE block IS one, and it is the last one:
W5e-2, [#2336](https://github.com/mudler/vllm.cpp/issues/2336). TWO further
entries had already left that list, and for the same reason in both cases: the seam is in `src/` and only the CALL is owed. The MoE weight
adapter onto `MoeBlockWeights` is no longer on it — W5d-4
([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4) is
`src/vllm/model_executor/models/qwen4_exp_moe.{h,cpp}`, which composes
`Qwen4ExpTextSparseMoeBlock` through the shared sparse-MoE seam rather than a
second MoE path, and the layer loop still has to CALL it. The mRoPE cos/sin
table build is not on it either — W5d-2
([#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 5) gave
`BuildMropeCosSinHost` external linkage behind
`include/vllm/model_executor/models/qwen3_5_mrope.h`, so the QSA half builds the
SAME tables the Qwen3.5/3.6 VL drivers build rather than a second copy — but the
loop still has to CALL it, and that call is W5b's.
The trap this paragraph used to warn about is FIXED, not pending: the loader
stores every gamma in the RAW HuggingFace parameterization and
`vt::Qwen4ExpGatedResidual` used to want the opposite, so a layer loop handing it
the loaded tensor applied a near-zero scale that reads as a checkpoint bug. W5b-6
([#2218](https://github.com/mudler/vllm.cpp/issues/2218)) moved the op onto the
loader's convention and gated the composition; a layer loop may now hand it
`Qwen4ExpWeights` directly. The
mixer/lm_head tail is not owed: the terminal
`use_combine=false` mixer IS `vt::Qwen4ExpGatedResidual` with a null
`block_inject`, gated as its own case in `test_qwen4_exp_hc_device.cpp`, and
`Qwen4ExpTextModel` has no final RMSNorm after it (`## Owed`), so the tail is
that op plus a `kMatmulBT` lm_head. W5c, the KV-cache spec with three conv states
and the QSA side cache — and note that this tree's runner accepts exactly ONE
`MambaSpec` group whose `shapes` must be exactly two, conv then temporal
(`src/vllm/v1/worker/gpu/runner.cpp`), so the third conv stream a PLE layer needs
is a runner change and not only a registry one
([#2131](https://github.com/mudler/vllm.cpp/issues/2131)); then the first served
request, G2 with a prompt past 2048 tokens, and only then the G4 speed axis,
which additionally waits on `dgx:gpu0`. MTP/speculators are W7
([#1993](https://github.com/mudler/vllm.cpp/issues/1993)). Each is scoped under
`## Owed` above with the structural facts W5a measured.

Both decisions this spec was blocked on are **settled** (developer, 2026-08-26) and
recorded in place rather than left as proposals: the transformers lane pin is
ACCEPTED at 5.16.0 (`## Oracles`), and the first runnable arm is the Q4_K_M backbone
with a non-resident n-gram table (`## Hardware`).

Next actions, in order: W2 (hashed n-gram embedding + PLE dilated depthwise conv) and
W3 (hyper-connection residual stream) are both reachable today against the lane pin
with tiny random configs and need neither a checkpoint nor a GPU lease — and both
inherit a config layer whose boundary is measured, so a golden that disagrees is a
port defect and not a config question. W6b's mechanism is the unknown that decides
whether the chosen arm is schedulable, and it should be spiked before W6 is planned.

## The sojufx sparkDash measurement (developer-directed, 2026-09-12)

The developer directed a speed comparison against
[sojufx/sojufx-Qwen3.8-Flash-Next](https://github.com/MiaAII-Lab/sparkDash) on
`dgx:gpu0` (NVIDIA GB10, `sm_121a`, aarch64, 128 GB unified memory). sojufx
publishes numbers from [sparkDash](https://github.com/MiaAII-Lab/sparkDash)
Decode Bench Structured: 400 output tokens, `temperature=0`, thinking off, at
concurrency 1/2/4. The benchmark targets an OpenAI-compatible
`/v1/chat/completions` endpoint with streaming.

**Protocol, reverse-engineered from the sparkDash source.** The prompt is
`"Count from 1 to 200. Output only the numbers, separated by spaces. No other
text."` The request body is `max_tokens=400, temperature=0, top_p=1,
stream=true, stream_options={include_usage: true}, min_tokens=400,
ignore_eos=true, stop=[], chat_template_kwargs={enable_thinking: false,
thinking: false, thinking_mode: "disabled"}`. Per-stream decode tok/s is
`(completion_tokens - 1) / (tLast - tFirst) * 1000`; aggregate decode tok/s is
`totalDecodeTokens / (max(tLast) - min(tFirst)) * 1000`.

### What each side ran

| | sojufx | vllm.cpp |
|---|---|---|
| Artifact | `Mia-AiLab/qwen3.8-Flash-Next-NVFP4` (~128 GB, NVFP4) | `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S (~68 GB) |
| Runtime | custom vLLM image, MTP speculative decoding (K=3), FP8 KV cache, 8-seq scheduler | `examples/vllm-server`, `--max-num-seqs 1`, no speculative decoding |
| KV cache | FP8 | f16, 256 blocks x 32 tokens = 8192 max context (auto-fit down from 262144); prefix caching disabled, async scheduling enabled at `max_concurrent_batches=2`. NOT PASSED, RESOLVED: neither launcher passed any of it -- both passed `--max-num-seqs 1 --device cuda --no-enable-thinking --enable-force-include-usage` and nothing else -- and the engine prints all three at boot. Re-read on 2026-09-12 on the same artifact, so these are defaults this configuration reproduces, not a recipe another run would have to repeat |
| Device | GB10 (`sm_121a`) | GB10 (`sm_121a`), same box |

### Result

| Concurrency | sojufx (tok/s) | vllm.cpp (tok/s) |
|---|---|---|
| C1 | 66.17 | 0.25 |
| C2 | 109.70 (aggregate) | not measured |
| C4 | 179.31 (aggregate) | not measured |

vllm.cpp C1: 99 decode tokens in 393.7 s, TTFT 9059 ms, model load ~1000 s
(~16.7 min) from the CIFS NAS. **The gap is ~264x at C1.** READ THE FIRST NUMBER
AGAIN BEFORE USING THE RATIO: 400 tokens were requested and 99 arrived, so this
is a truncated stream and not a completed benchmark, and the build it ran on was
a bare `cmake -DVLLM_CPP_CUDA_ARCHITECTURES=121a` whose compiled feature set was
never asserted. The ratio is quoted here as the thing that prompted the
re-measurement, not as a result this row stands behind. This is directional,
not a parity result: sojufx uses NVFP4 (~128 GB) with MTP speculative decoding
and 8-seq scheduling, while vllm.cpp uses UD-IQ1_S (~68 GB) with single-seq
decode. The comparison was published on the face as `TOKEN_GATE=FAIL` per the
developer's 2026-09-04 direction, because the token gate does not pass on this
architecture (3 of 6 disagreements vs llama.cpp free-running, per `## Owed`
[#2999](https://github.com/mudler/vllm.cpp/issues/2999)).

### The crash the benchmark found, and the fix that landed (#1958)

The benchmark crashed the server under `min_tokens` + `ignore_eos` before the
fix. `DeviceScratch` in `include/vllm/v1/sample/device_scratch.h` had a 0-copy
path on unified-memory backends (CPU, GB10): it wrapped the caller's host
pointer in-place. But `apply_min_tokens` and `apply_logit_bias` in
`src/vllm/v1/sample/logits_processor/builtin.cpp` pass function-local
`std::vector`s that are destroyed when the function returns, before the async
CUDA kernel reads them. On GB10 this produced `illegal memory access`. On
discrete GPUs the bug was masked because `~DeviceScratch` calls `cudaFree`,
which synchronizes the queue before the memory becomes invalid.

The fix (commit `1b2a49364`, landed on `main` 2026-09-12) removes the UMA
0-copy branch: `DeviceScratch` always allocates and copies on every backend.
The derived tensors are tiny (at most a few hundred int32s), so the copy is
negligible. Issue
[#1958](https://github.com/mudler/vllm.cpp/issues/1958) is closed; the fix is
owned by row `SAMPLE-CORE` (spec: `.agents/specs/sampling-controls-c7.md`).

### THE CONSUMER AUDIT FOR THE DEFERRED V-HEAD PERMUTATION, and the one place the "bit-identical" claim is too strong (2026-09-14)

Implementation of `ISSUE-LOCAL-01M2ENTH6YA5FWEDY6CFHF4NAM`. The scope named the
correctness question as the actual work: the load-time reorder exists so every
downstream consumer sees HuggingFace GROUPED V-head order, and permuting the
vectors moves that obligation outward. This is that audit, read off
`GdnBlockPaged`, `GdnBlock` and `GdnBlockPagedMixedSpec` rather than assumed.

**THE BINDING CONSTRAINT IS THE RECURRENCE'S HEAD PAIRING.** `vt::GdnDecode`,
`vt::GdnPrefill`, `vt::GdnSpecDecode` and `vt::GdnPackedDecode` take q/k at
`[T, Hk, Dk]` and v/g/beta at `[T, Hv, *]`, and they pair value head `v` with
key head `v / (Hv / Hk)`. That integer-divide rule IS the grouped order. In the
converter's tiled order the same pairing would read `v % Hk`, so running the
recurrence in tiled order is not an option, and the `attn_qkv` q and k rows the
converter does NOT tile are what makes it unavailable — there is no permutation
of 16 key heads that repairs a 48-head modulo grouping.

**SO THE PERMUTATION IS APPLIED AT THE PROJECTION BOUNDARY AND NOWHERE ELSE.**
Four sites, all before the first V-head-indexed consumer:

| vector | shape | permuted | first consumer, and why it is satisfied |
|---|---|---|---|
| `mixed`'s trailing V channels | `[T, conv_dim]` | tiled -> grouped, prefix `2*key_dim` | `vt::CausalConv1d{Fwd,Update,SpecUpdate}` is DEPTHWISE — channel `c` pairs with row `c` of `conv1d_weight`, which the loader still permutes at load. Both sides grouped. |
| `z` | `[T, value_dim]` | tiled -> grouped | `vt::RmsNormGated`'s gate, read as `[T, Hv, Dv]`. `norm_weight` is `[Dv]` and head-agnostic. |
| `a`, `b` | `[T, Hv]` | tiled -> grouped | `vt::GdnGBeta` / `vt::GdnPostConv` / `vt::GdnPackedDecode`, which pair them ELEMENTWISE with `a_log` and `dt_bias` — both still permuted at load. |
| the out-projection input | `[T, value_dim]` | grouped -> tiled | `out_proj`, whose permutation is on its COLUMNS, so the identity is `(W P) x = W (P x)` and the direction is the INVERSE one. |

**WHAT IS NOT PERMUTED AT RUN TIME, AND WHY THAT IS THE POINT.**
`ssm_conv1d`, `ssm_a` and `ssm_dt.bias` keep their load-time permutation. They
are f32 and tiny — 0 of the 4.152 GB — so deferring them would buy nothing and
cost a second permutation site. Leaving them grouped is exactly what confines
the run-time work to the table above: the causal conv, the delta rule, the gated
norm and BOTH PERSISTENT STATE CACHES (`conv_state` `[slots, conv_dim, K-1]`,
`ssm_state` `[slots, Hv, Dv, Dk]`) see the grouped order they have always seen,
so no cache written by an older build is re-indexed by a newer one.

`vt::IndexSelect` in `GdnBlockPagedMixedSpec` splits by TOKEN, never by channel,
and `vt::IndexCopy` merges the same way, so the mixed spec+prefill path inherits
the permutation from the projections it is handed and needs no site of its own.
Its out-projection tail needs the inverse, and has it.

#### THE BIT-IDENTITY CLAIM IS TRUE FOR FOUR OF THE FIVE AND OVERSTATED FOR `ssm_out`

The issue says "Both are exact re-indexings with no arithmetic, so both are
BIT-IDENTICAL". **The re-indexing is exact in both directions; the REDUCTION
that consumes it is not.**

- A ROW permutation moves whole dot products. Output element `g` is the same sum
  of the same products in the same order; only the index it is written to
  changes. `attn_qkv`, `attn_gate`, `ssm_beta` and `ssm_alpha` are bit-identical
  by construction.
- A COLUMN permutation permutes the summation order INSIDE every dot product,
  and floating-point addition is not associative. `ssm_out` is therefore
  bit-identical only under exact arithmetic. Whether it moves a float in
  practice is a measurement, and the `GDN deferred V-head permutation == the
  load-time one` case in `tests/vllm/models/test_gdn_v_head_permute.cpp` is the
  one that makes it.

**AND THAT MEASUREMENT IS NARROW ON TWO AXES, WHICH THE FIRST WRITING OF THIS
SECTION DISCLOSED ONLY ONE OF (2026-09-15 repair).** It reads `bad == 0`:

1. at `value_dim` **96**, not the released **6144** — the reordered dot product
   is 64x shorter than the one that ships, and the reassociation error a column
   permutation can introduce grows with the reduction length; and
2. on an `out_proj` built `[value_dim, H]` with `nk` UNSET, consumed by the
   **bf16 CPU `Matmul`** — while the released path is `[H, value_dim]` with
   `nk = true`, a **Q6_K** operand consumed by **`QuantDotGemm*`**. A different
   orientation, a different operand encoding and a different kernel each choose
   their own summation order, so none of the three is held fixed by this
   reading.

The released orientation is now covered on the VALUE axis by the deferred
subcase of `test_qwen4_exp_layer_loop.cpp` (`[H, value_dim]`, `nk = true`,
against the transformers golden at `kTol`), but that is a tolerance and not an
equality, so the bit-identity claim itself remains a 96-wide bf16-CPU reading.
Both comments that asserted it without qualification
(`qwen4_exp_weights.cpp:280-288`, `include/vt/ops.h:229`) were corrected in the
same repair; the commit body of `708a96589` is immutable and still carries the
unqualified wording.

**AND A SECOND, LARGER CAVEAT ON THE GATE AS THE ISSUE WROTE IT.** "Bit identity
against today's output on the released artifact" is not reachable by ANY
implementation of this change, because the change's whole purpose is to stop
dequantizing. Today the five projections are bf16 values rounded from an f32
dequant, consumed by cuBLAS `gemvx`; afterwards they are native Q5_K/Q6_K
operands consumed by `QuantDotGemm*`. Different operand encodings and a
different kernel cannot produce identical floats, and a gate that demanded it
would be demanding the defect back. What IS gateable at equality, and what the
committed tests gate, is the PERMUTATION MECHANISM held apart from the residency
change: the same weight values, in the same dtype, permuted at load versus
permuted on the vector.

#### THE COST IS O(T * N) PER LAYER PER FORWARD, AND ONE PREFILL STEP IN ISOLATION PAYS MORE THAN IT SAVES

The issue, `qwen4_exp_weights.cpp` and the commit body all said the permutation
is "O(n) on a vector of at most 10,240 elements". **That is a DECODE reading
with the token count dropped.** The op runs on `mixed [T, 10240]`,
`z [T, 6144]`, the out-projection input `[T, 6144]` and `a`/`b` `[T, 48]` each,
at five sites in `qwen3_5.cpp` (`:4536`, `:4539`, `:4741`, `:5251`, `:5775`),
once per layer per forward. Per token per layer that is 22,624 elements,
gathered read-and-written at bf16 = 90.5 kB, so **3.26 MB per token over 36
layers**.

The saving is **2.649 GB per STEP whatever T is** (4.152 GB of bf16 becoming
1.503 GB of file bytes). So:

| | permutation cost | saving | net |
|---|---|---|---|
| breakeven | 2.649 GB | 2.649 GB | T ~= **817** |
| one T = 2048 prefill step | 6.67 GB | 2.649 GB | **+4.0 GB, net-POSITIVE** |
| one decode step (T = 1) | 3.26 MB | 2.649 GB | **-2.646 GB** |

**READ ONLY THE SECOND ROW AND THE CHANGE LOOKS NET-NEGATIVE AT THE DEFAULT
TOKEN BUDGET. IT IS NOT, FOR ANY GENERATION LONGER THAN TWO TOKENS.** A
T = 2048 prefill overpays by 4.0 GB and each decode step underpays by 2.646 GB,
so the prefill is repaid after **1.5 decode steps**. End to end: a 2048-token
prompt plus 16 decodes is already **-38 GB**, and this row's reference workload
(35-token prompt, 400 decodes) is **-1060 GB**. The prefill loss is real and
worth stating; it is not a reason to fold the re-indexing into consumer
addressing, which is a different and much larger change that this arithmetic
does not justify.

**NEITHER HALF IS MEASURED ON A DEVICE.** The prefill cost, the decode saving
and the crossing point are all derived from byte counts. See `## Owed`.

#### THE END-TO-END SUITE WAS BLIND TO V-HEAD ORDER, AND THAT WAS THE MOST VALUABLE FINDING OF THE REVIEW

A fresh review inverted `vt::VHeadPermute`'s two index maps at every production
site and `test_qwen4_exp_layer_loop` stayed 14/14 and `test_qwen4_exp_forward`
2/2 **green**. The cause is not a degenerate fixture — the fixture's ratio is
K = 2 against R = 3, the released ratio in miniature. The cause is that
`FillGdn` (`test_qwen4_exp_layer_loop.cpp`) builds `Qwen4ExpGdnWeights` by hand
and never sets `v_head_perm_key_heads`, so the deferred permutation was **inert
in the one transformers-oracle case in the tree**, and the whole out-projection
permutation rested on one assertion, `CHECK(bad == 0)` in
`test_gdn_v_head_permute.cpp`.

The repair adds a second ARM to that oracle case rather than a second case:
`TileGdnVHeads` re-indexes the same values into the converter's tiled V-head
order on exactly the five projections, sets the flag, and the arm is gated
against the SAME transformers 5.16.0 golden at the same bound. It also runs the
RELEASED `out_proj` orientation — `[H, value_dim]` with `nk = true` — which
`test_gdn_v_head_permute.cpp` does not.

Three smaller test defects were repaired with it:

- `MakeTiledWeights`'s `which == 2` flipped `in_proj_b` AND `in_proj_a`
  together, so neither was ever convicted alone. They are indices 2 and 3 now,
  and `out_proj` moved to 4; the red-first loop runs five cases.
- `RunLayer` took `ssm` and `conv` BY VALUE, so the block's writes to both
  persistent caches died with the call and the spec's "both persistent state
  caches see the grouped order they have always seen" claim had no test. It
  returns them now, and the equality case asserts both are bit-equal across the
  two arms AND that both actually moved.
- The two axes on which the `bad == 0` reading is narrow are stated where the
  claim is made, in the test and in both shipped comments.

### THE NEXT ROW, NAMED: keep the GDN projections quantized by permuting a vector (2026-09-14)

The attribution found that **61% of decode weight traffic is a load-time bf16
expansion**: the V-head reorder the released 16-vs-48 config forces cannot be
applied to a k-quant block stream, so `attn_qkv`, `attn_gate`, `ssm_beta`,
`ssm_alpha` and `ssm_out` of all 36 GDN layers are dequantized at load --
**4.152 GB/step against 1.508 GB in the file**. `qwen4_exp_weights.cpp:292-302`
states the cost against itself.

**The fix that comment does not consider is to permute a VECTOR instead of the
weight.** For a GEMV, a ROW permutation satisfies `(P W) x = P (W x)` (permute
the output) and a COLUMN permutation satisfies `(W P) x = W (P x)` (permute the
input). Both are exact re-indexings, so both are BIT-IDENTICAL, and both are O(n)
on a vector of at most 10,240 elements against 115 MB of weight traffic per
layer. **This covers `ssm_out`'s column reorder**, the case the comment calls
unavailable -- because it only considered permuting the weight.

| | bytes/step | floor at 273 GB/s | |
|---|---|---|---|
| today | 6.792 GB | 24.88 ms | ~40 tok/s |
| ROW reorders only (73% of the expansion) | ~4.81 GB | 17.6 ms | ~57 tok/s |
| **all five** | **4.148 GB** | **15.19 ms** | **~66 tok/s** |

**THE ONE WAY THIS DISAPPOINTS, and it must be measured rather than assumed:**
freeing the bytes moves these operands from cuBLAS bf16 GEMV, which achieves
162.3 GB/s here, to our own `QuantDotGemm*` family, which achieves 93.4 GB/s. A
1.7x lower achieved rate against a 2.75x lower byte count is still a win, but the
margin is smaller than the byte ratio suggests and the trade is the thing to
measure first.

Scoped in `ISSUE-LOCAL-01M2ENTH6YA5FWEDY6CFHF4NAM`, which also lists the
correctness risk that is the actual work: the reorder exists so every downstream
consumer sees HF V-head order, and permuting vectors moves that obligation to the
conv, the delta rule and the gate.

### THE GAP IS A 13x MEMORY-EFFICIENCY DEFICIT, NOT A CEILING (2026-09-13)

Derived from the model's config and this row's measured step, no GPU needed.
**We sustain ~11.0 GB/s of GB10's ~273 GB/s: 4.0% of peak.** sojufx at 66.17
tok/s sustains 140.1 GB/s (51.3%), and **this tree already sustains 220.8 GB/s
(81%) on another model on this same box.**

Active weights are 4.231 G params per token (48 layers x 88.15 M: MoE top-10 of
512 at `moe_intermediate_size` 640 plus a shared expert, plus q/kv/o), which at
IQ1_S is **~0.85 GB per step**. At 273 GB/s the floor is **3.1 ms per step,
~320 tok/s**. The reference's 15.1 ms step is five times slower than the memory
system permits, and at even 50% of peak we would be at ~161 tok/s -- 2.4x PAST
the reference.

**So the 5.08x gap is a defect, not a limit, and two of this row's own
conclusions today are corrected by it.** The withdrawn "no combination of the
identified kernel leads reaches 66" was wrong in a second and more basic way than
its first retraction: it treated the distance as a shortfall to scrape together
when there is ~20x of headroom. And the MTP finding
(`ISSUE-LOCAL-01M2EG6R3MRCB9ENZB9840KAXX`), though true and unretracted, was
OVER-WEIGHTED as "the blocker": speculation multiplies throughput at a given step
cost and cannot explain running at 4% of bandwidth.

Owed, and it is the next row: a per-kernel bandwidth attribution over the 68.9%
that is GEMM, from shapes rather than from `ncu` (which refuses here). See
`ISSUE-LOCAL-01M2EK69SESGH6ST1ESMFZC808`, which also records why the q/k/v merge
is NOT the place to start.

### THE PER-KERNEL BANDWIDTH ATTRIBUTION, and it FALSIFIES the 4%-of-peak figure above (2026-09-14)

Derived from `tests/vllm/models/qwen4_exp_gguf_manifest.inc` (the committed
header manifest of the released UD-IQ1_S artifact: every tensor's ggml type id
and dims), the loader's residency rules, and the measured kernel table below.
**No GPU, no lease and no `ncu`** -- which is the point, because `ncu` refuses on
this fleet with `ERR_NVGPUCTRPERM`.

**THE HEADLINE IS A CORRECTION, AND THE SECTION IT CORRECTS IS THE ONE DIRECTLY
ABOVE.** That section says the step moves **~0.85 GB** and therefore sustains
**11.0 GB/s, 4.0% of peak**. Read off the artifact's own type ids, the decode
step moves **6.792 GB of GEMM weight operands**, which is **8.0x** its figure,
and the step therefore sustains **88.3 GB/s against 76.9 ms of kernel time:
32.4% of GB10's ~273 GB/s**. Nothing in the decode path is at 1-5% of peak. The
lowest measured row is at 15%.

#### Where the 0.85 GB went wrong, in four parts

| | the section above | the manifest |
|---|---|---|
| active GEMM params/token | 4.231 G | **6.625 G** |
| average width | IQ1_S, ~1.6 bit | **8.20 bit** |
| bytes/step | ~0.85 GB | **6.792 GB** |

1. **It priced the whole model at IQ1_S.** IQ1_S is 68 of the 96 routed expert
   gate/up towers and NOTHING else. `ffn_down_exps` is IQ4_NL (4.5 bit), the
   remaining 28 gate/up towers are IQ2_XXS, the shared experts are Q5_K/Q6_K/Q8_0,
   the hyper-connections are Q8_0 (8.5 bit), and `output.weight` is Q4_K.
   The artifact's whole type histogram, from the manifest, is **557 F32, 244 Q8_0,
   212 Q5_K, 68 IQ1_S, 49 IQ4_NL, 40 Q6_K, 28 IQ2_XXS, 24 BF16, 2 Q4_K** over
   1,224 tensors. There is **no Q5_0 in this file at all**, which is worth saying
   because this row's briefing prose has described the expert towers as
   "IQ4_NL/Q5_0": down is IQ4_NL, gate/up are IQ1_S and IQ2_XXS, and the 49th
   IQ4_NL tensor is `per_layer_token_embd`.
2. **It omitted whole stages.** The hyper-connections are **668.4 MB/step** (four
   Q8_0 `[10240,320]`/`[320,10240]` projections in every one of 48 layers), the
   `lm_head` is **357.6 MB**, and the MoE router is **125.8 MB**. None appears in
   its arithmetic.
3. **It gave all 48 layers the same attention.** 36 are Gated DeltaNet with much
   wider projections and 12 are QSA.
4. **AND THE LARGEST TERM IS NOT IN THE FILE AT ALL.** See the next subsection.

#### 61% OF THE STEP'S TRAFFIC IS A LOAD-TIME bf16 EXPANSION

`qwen4_exp_weights.cpp:277-395` reorders the V heads of every Gated DeltaNet
projection from HF's grouped order into ggml's tiled order, and a k-quant
superblock spans elements that permutation moves. So on the released
`linear_num_key_heads` 16 vs `linear_num_value_heads` 48 config the reorder is
ALWAYS on, and `attn_qkv`, `attn_gate` and `ssm_out` of all 36 linear layers
**dequantize to bf16 at load** (`DequantAll` then `Bf16From`, three call sites,
no keep-quant arm). The file's own comment at `:300-302` states this and is
correct; what has never been written down is its size.

```
  36 layers x (52.43 + 31.46 + 31.46) MB bf16   =  4.152 GB / step
  the same tensors in their file encoding        =  1.508 GB / step
                                                    -------  2.75x
```

**That single loader property is 61% of every decode step's weight traffic**, and
it is also why cuBLAS appears at all in a keep-quant model: an elementwise bf16
weight leaves `vt::MatmulBT`'s block-quant branch (`src/vt/ops.cpp:192-195`) and
lands in cuBLAS `gemvx`, while every block-quant weight goes to `QuantDotGemm*`.
The 34.8%/34.1% split in the kernel table IS that dtype split.

#### The attribution, ranked by achieved bandwidth ASCENDING

Only four rows of the capture carry a committed average duration; see
"What is NOT attributed" below.

| kernel | call site | weight | K x N | dtype | MB/call | calls/step | measured avg | achieved | % of 273 |
|---|---|---|---|---|---|---|---|---|---|
| `QuantDotGemmGroupedKernel` | `qwen4_exp_moe.cpp:190` | `ffn_gate_exps`/`ffn_up_exps`, 10 of 512 | 2560x640 | IQ1_S | 3.200 | 68 | 77 us (1) | **41.6 GB/s** | **15.2%** |
| `QuantDotGemmGrouped32Kernel` | `qwen4_exp_moe.cpp:190` | `ffn_down_exps`, 10 of 512 | 640x2560 | IQ4_NL | 9.216 | 48 | 163 us (1) | **56.5 GB/s** | **20.7%** |
| cuBLAS `internal::gemvx::kernel` #1 | GDN in/out projections | two of `attn_qkv`/`attn_gate`/`ssm_out` | see (2) | bf16, EXPANDED | 31.46-41.94 | 71.9 | 244 us | **129-172 GB/s** | **47-63%** |
| `QuantDotGemmKernel<WType 4, float>` | `qwen4_exp_registry.cpp:1032` | `output.weight` | 2560x248320 | Q4_K | 357.581 | 1.0 | 2.49 ms | **143.6 GB/s** | **52.6%** |

(1) These two averages come from the **~1600-token capture** at `ee0644eab`, not
from the 400/600-token capture the rest of this section uses. Instance counts are
context-free (one grouped call per tower per layer) but the durations are not
guaranteed to transfer. Treat them as indicative and re-read them from the
retained `4e36bbae` capture.

(2) 244 us x 71.9 calls = 17.53 ms, and the whole cuBLAS group is 26.76 ms for
4.343 GB. **The small-weight readings are therefore impossible and are excluded
by arithmetic, not by preference**: if row #1 were `ssm_alpha`+`ssm_beta`
(72 calls, 0.246 MB each) the remaining 4.15 GB would have to move in 9.23 ms,
i.e. at 450 GB/s, above the machine. Row #1 is two of the three large GDN
projections. WHICH two the committed record does not say, so the row is given as
a range: 31.46 MB/call if it is `attn_gate`+`ssm_out`, 41.94 MB/call if it pairs
`attn_qkv` with one of them.

#### Group level, which is where the answer actually is

| family | bytes/step | measured | achieved | % of 273 |
|---|---|---|---|---|
| cuBLAS `gemvx` + `gemv2T` + `splitKreduce` (elementwise bf16) | 4.343 GB | 26.76 ms | **162.3 GB/s** | **59.5%** |
| our `QuantDotGemm*` (block-quant) | 2.449 GB | 26.22 ms | **93.4 GB/s** | **34.2%** |
| all GEMM/GEMV | 6.792 GB | 52.98 ms | 128.2 GB/s | 47.0% |
| the whole step, kernel time | 6.792 GB | 76.9 ms | **88.3 GB/s** | **32.4%** |
| the whole step, wall | 6.792 GB | 87.6 ms | 77.5 GB/s | 28.4% |

**So the ranking the next row needs is the opposite of the one it was going to
inherit.** cuBLAS is the larger half of the GEMM bill because it moves 1.8x the
bytes, not because it is slow: at 59.5% of peak it is the MORE efficient half.
Our own grouped k-quant kernels are the least efficient measured rows at 15-21%.

#### THE FLOOR IS NOT 320 tok/s AND THE HEADROOM IS NOT 20x

| | bytes/step | floor at 273 GB/s | implied ceiling |
|---|---|---|---|
| the section above | 0.85 GB | 3.1 ms | 320 tok/s |
| **as this engine loads it today** | **6.792 GB** | **24.88 ms** | **40 tok/s** |
| **with the GDN expansion removed** | **4.148 GB** | **15.19 ms** | **66 tok/s** |

The middle row is the one that changes a plan: **at 6.792 GB per step the memory
system cannot deliver 66 tok/s on this artifact at all**, so the GDN keep-quant
repair is not an optimisation among others, it is the precondition for the target
being reachable. And the bottom row lands on **66 tok/s**, against sojufx's
measured 66.17. That is a coincidence of rounding and not a derivation of the
reference's rate -- sojufx runs a ~4-bit NVFP4 checkpoint with a different active
set -- but it does say the reference is operating AT the memory floor of an
equivalently-sized artifact, and we are at 3.1x of ours rather than 20x.

#### The full byte inventory, per call site, per step

Every GEMM operand the decode path reads, from the manifest. `calls/step` is
derived from the architecture (48 layers, 36 GDN + 12 QSA, MoE top-10 of 512 plus
a shared expert) and cross-checks against the capture wherever the capture names
an instance count.

| call site | K x N | dtype | MB/call | calls/step | MB/step | served by |
|---|---|---|---|---|---|---|
| GDN `in_proj_qkv` (`attn_qkv`) | 2560x10240 | bf16 EXPANDED from Q5_K/Q6_K | 52.429 | 36 | **1887.4** | cuBLAS |
| GDN `in_proj_z` (`attn_gate`) | 2560x6144 | bf16 EXPANDED | 31.457 | 36 | **1132.5** | cuBLAS |
| GDN `out_proj` (`ssm_out`) | 6144x2560 | bf16 EXPANDED from Q6_K | 31.457 | 36 | **1132.5** | cuBLAS |
| MoE `ffn_down_exps` x10 | 640x2560 | IQ4_NL | 9.216 | 48 | 442.4 | `QuantDotGemmGrouped32Kernel` |
| `lm_head` `output.weight` | 2560x248320 | Q4_K | 357.581 | 1 | 357.6 | `QuantDotGemmKernel<4,float>` |
| QSA q+gate (`attn_q`) | 2560x12288 | Q5_K | 21.627 | 12 | 259.5 | `QuantDotGemm*` Q5_K |
| hyper-connection `hc_attn_down` | 10240x320 | Q8_0 | 3.482 | 48 | 167.1 | `QuantDotGemmQ8_0Kernel` |
| hyper-connection `hc_attn_up` | 320x10240 | Q8_0 | 3.482 | 48 | 167.1 | `QuantDotGemmQ8_0Kernel` |
| hyper-connection `hc_ffn_down` | 10240x320 | Q8_0 | 3.482 | 48 | 167.1 | `QuantDotGemmQ8_0Kernel` |
| hyper-connection `hc_ffn_up` | 320x10240 | Q8_0 | 3.482 | 48 | 167.1 | `QuantDotGemmQ8_0Kernel` |
| QSA `o_proj` (`attn_output`) | 6144x2560 | Q5_K | 10.813 | 12 | 129.8 | `QuantDotGemm*` Q5_K |
| MoE router `ffn_gate_inp` | 2560x512 | bf16 EXPANDED from F32 | 2.621 | 48 | 125.8 | cuBLAS |
| MoE `ffn_gate_exps` | 2560x640 | IQ1_S | 3.200 | 34 | 108.8 | `QuantDotGemmGroupedKernel` |
| MoE `ffn_up_exps` | 2560x640 | IQ1_S | 3.200 | 34 | 108.8 | `QuantDotGemmGroupedKernel` |
| shared-expert down | 640x2560 | Q8_0 | 1.741 | 48 | 83.6 | `QuantDotGemmQ8_0Kernel` |
| MoE `ffn_gate_exps` | 2560x640 | IQ2_XXS | 4.224 | 14 | 59.1 | `QuantDotGemm*` IQ2_XXS |
| MoE `ffn_up_exps` | 2560x640 | IQ2_XXS | 4.224 | 14 | 59.1 | `QuantDotGemm*` IQ2_XXS |
| shared-expert gate | 2560x640 | Q5_K x34 / Q6_K x14 | 1.126-1.344 | 48 | 54.3 | `QuantDotGemm*` |
| shared-expert up | 2560x640 | Q5_K x34 / Q6_K x14 | 1.126-1.344 | 48 | 54.3 | `QuantDotGemm*` |
| QSA indexer `q_proj` | 2560x512 | BF16 in file | 2.621 | 12 | 31.5 | cuBLAS |
| PLE `ple_key` | 2560x10240 | Q8_0 | 27.853 | 1 (3) | 27.9 | `QuantDotGemmQ8_0Kernel` |
| GDN `in_proj_a`/`in_proj_b` | 2560x48 | bf16 EXPANDED from F32 | 0.246 | 72 | 17.7 | cuBLAS |
| QSA `k_proj` + `v_proj` | 2560x512 | Q5_K | 0.901 | 24 | 21.6 | `QuantDotGemm*` Q5_K |
| QSA indexer `k_proj` | 2560x128 | BF16 in file | 0.655 | 12 | 7.9 | cuBLAS |
| PLE `ple_value` | 2560x2560 | Q8_0 | 6.963 | 1 (3) | 7.0 | `QuantDotGemmQ8_0Kernel` |
| final `output_hc_down`/`_up` | 10240x320 | Q8_0 | 3.482 | 2 | 7.0 | `QuantDotGemmQ8_0Kernel` |
| `hc_attn_inject`/`hc_ffn_inject` | 10240x4 | bf16 EXPANDED from F32 | 0.082 | 96 (4) | 7.9 | cuBLAS |
| **TOTAL** | | | | | **6792** | |

(3) `QuantDotGemmQ8_0Kernel`'s measured **242.0** calls/step is 192 hyper-connection
+ 48 shared-expert-down + the 2 final-block projections, with NO room for PLE's
two Q8_0 GEMMs. So the count is evidence that **the PLE block does not run on a
decode step**; if that reading is right the total is 6.757 GB and every ratio in
this section moves by 0.5%. The rows are kept in the inventory because the
alternative pairing (PLE in, final block out) also sums to 242 and the capture
cannot separate them.
(4) `hc_*_inject` IS a `LoadMatmul` operand (`qwen4_exp_weights.cpp:270-273`),
N = `hc_count` 4, so it is a real GEMV. 0.12% of the total.

#### THE COUNTS CROSS-CHECK, and that is what makes the mapping more than a guess

Three instance counts in the capture are matched EXACTLY by the architecture, and
none of the three was fitted:

| capture | per step | architecture | matches |
|---|---|---|---|
| `QuantDotGemmQ8_0Kernel` 145,200 / 600 | **242.0** | 4 hc x 48 + shexp-down x 48 + 2 | **242** |
| `gemvx` #1 43,128 / 600 | **71.9** | two of three GDN dense projections x 36 | **72** |
| `QuantDotGemmGroupedKernel` 35,816 / 527 | **67.96** | IQ1_S gate/up towers, 34 layers x 2 | **68** |
| `QuantDotGemmGrouped32Kernel` 25,277 / 527 | **47.96** | `ffn_down_exps`, one grouped call per layer | **48** |
| `QuantDotGemmKernel<WType 4, float>` 600 / 600 | **1.0** | `output.weight`, the ONLY Q4_K GEMM operand | **1** |

The 68 is the sharper one. 68 is not a divisor of 48, 36 or 12, and it is not a
count of layers at all: it is the number of expert gate/up TOWERS whose file type
is IQ1_S (34 layers x 2 towers), the other 28 being IQ2_XXS and landing in a
different template instantiation. It also confirms, independently of any grep,
that gate and up are issued as SEPARATE grouped calls -- this model does not use
`layers::MlpGateUpMethodBase` or `vt::MergedGemmGroup`, exactly as the seam
section above says.

#### What is NOT attributed, and what would close it

- **Ten of the roughly fourteen GEMM rows have no committed average duration.**
  Only the group totals (26.76 ms and 26.22 ms) and four individual rows were
  ever written down; the full 25-row `nsys stats` output was not committed. The
  capture `4e36bbae` is retained, so re-running `nsys stats` on it closes this
  with a two-minute lease and no model load, exactly as `fe5be663` did.
- **Which two of the three large GDN projections form `gemvx` row #1**, hence the
  129-172 GB/s range on that row.
- **The two grouped-kernel durations are cross-capture** (note (1)).
- **`QuantDotGemmQ8_0Kernel`'s 242 resolves two ways**: 192 hc + 48 shared-expert
  down + the 2 final-block hc projections, or the same with PLE's 2 calls instead
  of the final block's. The difference is 28 MB/step, 0.4%.
- **Whether the PLE block runs on every decode step.** Note (3) argues from the
  242 count that it does not. 34.9 MB/step, 0.5%, and it changes no conclusion.
- **KV-cache traffic is excluded.** This is the weight budget only.
  `QsaGatherAttentionKernel`'s 18.15 ms is not a weight read and is not in any
  ratio above.
- **Activation traffic is excluded**, which is correct at batch 1: the activation
  is ~5 KB against tens of MB of weights per call, under 0.1%.

#### What this row should do next, in this order

1. **Keep the Gated DeltaNet projections quantized.** 2.644 GB/step, 39% of all
   traffic, and the only item that moves the floor from 40 tok/s to 66. The
   reorder is the obstacle; it is a V-head PERMUTATION of whole rows for
   `attn_qkv` and `attn_gate`, which a block stream can express as a row
   permutation, and a COLUMN permutation for `ssm_out`, which it cannot
   (`qwen4_exp_weights.cpp:300-302` already says so). So two of the three are
   probably recoverable and the third needs its own answer.
2. **Then the grouped k-quant kernels at 15-21% of peak**, which is the only
   genuinely low efficiency the attribution found, over 778 MB/step.
3. **Do NOT start from the q/k/v merge**, for the reason the issue already gives
   and which this attribution now quantifies: at batch 1 the activation is under
   0.1% of the bytes a GEMV moves.

### THE SLOWEST KERNEL IS ONE WARP PER OUTPUT WITH A 2.5-ITERATION LOOP (2026-09-18)

The bandwidth attribution ranked the decode kernels ascending and the floor is
the IQ1_S expert gate/up grouped kernel at **41.6 GB/s, 15.2% of peak**, against
cuBLAS `gemvx` at 162.3 GB/s in the same capture.

Read off the source: `cuda_quant_dot.cu:1986` `QuantDotGemmGrouped32Kernel`
assigns **one warp per output element**, and `LaunchGrouped32` (`:2092`) uses
`kWarpsPerBlock = 4`. At `hidden_size` 2560 the weight row is `nb = 80` blocks,
so `for (b = lane; b < 80; b += 32)` is **2.50 iterations per lane** -- two or
three loads, then a five-step `__shfl_down_sync`, then 31 of 32 lanes idle while
lane 0 writes one float. Five reduction steps to amortise two and a half loads.
That is latency-bound by construction.

**The fix has a precedent 120 lines below it.**
`QuantDotGemmGroupedFusedSwiGLU32Kernel` (`:2019`) already amortises at width 2 --
one warp computes gate AND up for the same `(p, j)` against one broadcast
activation. Generalising that to N output columns per warp makes the loop
`2.5 x N` useful iterations per reduction and reads the broadcast activation once
per warp instead of once per output.

**Bound, not promise:** at cuBLAS's measured 162.3 GB/s the family's 26.22 ms/step
becomes ~15.1 ms, about 11 ms off a 76.9 ms step; at full peak, ~17 ms. Neither is
predicted -- 15.2% of peak has one measurement behind it, and register pressure
turns the N sweep over somewhere. Scoped in
`ISSUE-LOCAL-01M2TZ9AE416FW4GH23V09783Q`, which owes the sweep before any width
is chosen.

**And the bit-identity bar here IS achievable**, unlike the two this row has
already written and had to retract: each output keeps its own accumulator and its
own ascending block order, so only the assignment of work to warps changes.

### THE FULL DECODE KERNEL TABLE, and a retraction: the lead is GEMM, not QSA (2026-09-13)

Read from the SAME self-validated capture as the allocator measurement
(`rc` job `4e36bbae`, 1,594,621 `cudaLaunchKernel` calls, 113 MB), by re-running
`nsys stats` on it in a 2-minute lease with no build and no model load
(`rc` job `fe5be663`). Per step over the window's 600 decode steps, against
76.9 ms of GPU kernel time:

| group | share | per step |
|---|---|---|
| **GEMM / GEMV, ALL OF IT** | **68.9%** | **52.98 ms** |
| -- cuBLAS `gemvx` family (4 rows) + `gemv2T` + `splitKreduce` | 34.8% | 26.76 ms |
| -- our `QuantDotGemm*` kernels (8 rows) | 34.1% | 26.22 ms |
| `QsaGatherAttentionKernel` | 23.6% | 18.15 ms |
| everything else (18 rows) | 7.5% | 5.77 ms |

Notable individual rows: `gemvx` #1 at 43,128 instances (71.9 a step) and 244 us
each; `QuantDotGemmQ8_0Kernel` at 145,200 instances (242 a step); and
`QuantDotGemmKernel<WType 4, float>` at exactly **600 instances -- one per step --
costing 2.49 ms each**, which is the shape of an `lm_head` over a large vocab.

#### RETRACTION

This row stated twice, in the MTP section and in the session prose that preceded
it, that **"no combination of the identified kernel leads reaches 66 on this
artifact without speculation."** That sentence was written while only TWO of the
twenty-five kernel rows had ever been printed -- `QsaGatherAttentionKernel` and
the first `gemvx` -- with the other 41.2 ms of the step unattributed. **It is
withdrawn as reasoning**, independently of how the measurement below turns out.
The arithmetic it quoted (deleting the top two kernels leaves ~52 ms) is correct
and is not withdrawn; what is withdrawn is the inference that those two WERE the
leads, and the closure that followed from it.

It also corrects this row's aim. W9 optimised QSA because the post-W7 re-rank put
it first. **At the reference workload QSA is second, at a third of the GEMM
bill.**

#### THE UNTRIED LEVER THIS EXPOSES, and it is already measured on this box

`.agents/` and this project's own records carry the finding, measured on
`dgx:gpu0` at tree `ecf580ce9`: **every decode GEMM runs a `cutlass_80_*` --
AMPERE -- tactic on a Blackwell `sm_121` part, with no `nvjet`, `sm120`, `sm121`
or `tcgen05` kernel anywhere in the decode trace**, alongside the sibling finding
that cuBLASLt enumerates no Blackwell algorithm for these descriptors and that
the DESCRIPTOR is the lever. That applies to the 26.76 ms/step of cuBLAS GEMV
above. The other 26.22 ms/step is OUR OWN hand-written quant GEMM kernels, which
no such finding covers and which nothing has profiled at this workload.

**WHAT IS NOT ESTABLISHED, and must not be assumed from the paragraph above.**
The recorded Ampere-tactic finding was measured on a DIFFERENT model and tree,
and on GEMM shapes; the rows here are `gemvx`, cuBLAS's GEMV path, which at
batch 1 is a different regime and may be bandwidth- rather than tactic-bound.
Whether a Blackwell tactic is reachable for THESE descriptors and these shapes,
and what it would be worth, is unmeasured. `ncu` refuses on this container
(`ERR_NVGPUCTRPERM`), so the achieved-bandwidth reading that would settle the
regime question is not available by the obvious route.

#### AND THE FIRST LEVER IS A SHARED SEAM THIS MODEL NEVER JOINED

`AGENTS.md` "Shared seams" requires: *"Route model fusion through
`vt::FusedChain`. Route mergeable multilayer perceptron (MLP) projections through
`layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`."*

**`qwen4_exp` uses none of the three.** Grepping every `qwen4_exp*.cpp/h` for
`MergedGemmGroup`, `MlpGateUpMethodBase` and `FusedChain` returns nothing; the
only users in the tree are `dots3_note_vision.h` and `glm5_next_bridge.h`.

The consequence is visible in the block code and then in the profile.
`qwen4_exp_qsa_block.cpp` issues the projections as SEPARATE `vt::MatmulBT`
calls against the same `hidden`: the indexer's q and k at `:593` and `:612`, then
`:692` (q gate), `:756` (`k_proj`) and `:764` (`v_proj`). Each one re-reads the
same activation and launches its own GEMV. The capture counts roughly **275 GEMV
launches per decode step** across the `gemvx`, `gemv2T` and `QuantDotGemm*`
families.

Merging q/k/v into one GEMM is the standard form of this: three launches become
one, the activation is read once instead of three times, and the single larger
GEMM has more work to hide latency behind -- which matters precisely because
these are BATCH-1 GEMV shapes, where per-launch overhead and activation re-reads
are a large fraction of a small kernel. This is a protocol requirement this model
does not meet AND the largest structurally-obvious lever aimed at the 68.9%.

**IT IS NOT SIZED HERE.** How much of the 52.98 ms/step a merge recovers depends
on how much of each GEMV is activation traffic against weight traffic, which this
capture does not separate. That is what the owed work measures.

**Owed, in order:** (1) confirm the tactic names actually chosen for these
descriptors at this workload, which `nsys` can report without `ncu` -- note the
rows here are `internal::gemvx::kernel`, cuBLAS's GEMV path, NOT the
`cutlass_80_*` GEMM tactics the recorded Blackwell finding measured, so that
finding may not transfer; (2) size the q/k/v and gate/up merge against the seam
`AGENTS.md` already mandates, which needs no new kernel and no new oracle; (3)
separately profile our own `QuantDotGemm*` family, the same size as the cuBLAS
half and carrying no prior finding at all. Do not scope a fix before (1) and (2)
are measured, and do not assume the Blackwell record applies to a GEMV.

### THE ALLOCATOR, MEASURED AT THE REFERENCE WORKLOAD: the count is right and the LEVER IS NOT (2026-09-13)

`dgx:gpu0`, `rc` job `4e36bbae`, source `3eabd4dbd` (post-W9 `main`), released
UD-IQ1_S, 600-token decode, `nsys` window opened and closed BY THE CLIENT around
the measured request. **The window is self-validated**: it recorded **1,594,621**
`cudaLaunchKernel` calls, and the job refuses to report numbers below 20,000
because a load-shaped window reads exactly like a result. The capture is 113 MB
against the 198 KB of the attempt that silently profiled model loading.

#### The prediction held to 2.1%

`ISSUE-LOCAL-01M2DQHP2FWXHH7GTHB17QX53Q` derived 97 `cudaFree` per step from
three call sites -- `Qwen4ExpGatedResidual` twice per layer plus once after, at
48 layers -- by reading the tree, and called it an arithmetic match needing a
measurement.

| | derived | measured |
|---|---|---|
| `cudaFree` per step | 97 | **99.0** (59,400 / 600) |
| `cudaMalloc` per step | 97 | **99.0** (59,428), paired to **28 calls** across the window |
| `cudaLaunchKernel` per step | ~2,560 | **2,658** |
| QSA launches per step | 12 | **12.0** (7,200 / 600) -- exactly the 12 QSA layers |

`cudaFree` is **80.0% of all CUDA API time**, mean **652.5 us**, i.e. **64.6 ms
per step** against an **87.6 ms** step: **74% of the step, in host time.**

#### AND IT IS WORTH ABOUT 14%, NOT 4x, BECAUSE THE GPU IS 88% BUSY

This is the number the issue said would size the work, and it changes the answer.
QSA is 23.6% of GPU kernel time at 10.895 s, so total kernel time in the window is
46.16 s -- **76.9 ms of the 87.6 ms step. The GPU is BUSY 88% of the time and
IDLE 10.6 ms per step.**

So the 64.6 ms of `cudaFree` **overlaps GPU work almost entirely**. Removing it
cannot remove work that was never on the critical path; it can only recover the
idle:

```
  best case   87.6 ms -> 76.9 ms      11.4 -> 13.0 tok/s      +14%
```

**THE BOUND THIS ROW WROTE BEFORE MEASURING IS WHAT MAKES THAT LEGIBLE.** The
issue said the recoverable time "is bounded above by how long the GPU is currently
IDLE, and not by the host-side total", and that is exactly what happened. A
reader given only "74% of the step is `cudaFree`" would have scoped a
caller-supplied workspace redesign, with its signature change and its re-entrancy
argument and its collision with `ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ`'s
retention finding, for a 14% return.

**The fix is still worth doing and is still cheap** -- `cudaFreeAsync` sits in
this very trace at **1,340 ns** against `cudaFree`'s **652,550 ns**, 487x -- but
it is a 14% item competing with 20%+ items, not the top of the list.

#### WHAT THE STEP IS ACTUALLY MADE OF, and it is GPU work

| kernel | share of kernel time | per step |
|---|---|---|
| `QsaGatherAttentionKernel` | **23.6%** | **18.16 ms** |
| cuBLAS `gemvx` (bf16) | **22.8%** | **17.53 ms** |

**We are GPU-BOUND at 88%.** The path to the reference is LESS GPU WORK, not less
host overhead, and that re-ranks this row for the third time today.

**ONE NEW OBSERVATION, NOT YET A SCOPE.** QSA is still 20.7% of the step AFTER W9,
at **1.513 ms per launch**. Against the pre-W9 profile's 2.545 ms at `|sel| ~
1600`, a cost linear in `|sel|` would predict roughly 0.53 ms at this run's mean
`|sel|` of about 335. It measures 1.513. **That implies a large per-launch cost
that does not scale with context at all**, which every context-scaling argument in
this row -- including the correction section above -- has been silently treating as
zero. Twelve launches a step makes it the larger half of QSA's bill here. The
owed instrument is a `|sel|` sweep at fixed shape on this box, which is the same
head_dim-sweep method the W9 attribution already used and trusts.

**THE SWEEP WAS ATTEMPTED TWICE AND BOTH RUNS DIED THE SAME WAY. Read this before
taking a lease for it.** `rc` jobs `8ea56726` and `0cfb7294` extended the working
single-window harness to two legs -- 120 tokens then 1200, one boot, one binary,
`nsys start`/`stop` around each. In both runs the server came up, answered
`/v1/models`, and SERVED THE WARMUP CORRECTLY (ttft 111.2 s and 73.1 s
respectively, decode 12.01 tok/s), and then **both legs got
`URLError: [Errno 111] Connection refused`** -- including the FIRST leg, before
any second `nsys start`, so it is not the two-window structure. The server died
between the warmup completing and the first capture. `nsys stop` still produced
`.nsys-rep` files, and the self-validation correctly REFUSED both as not
decode-shaped rather than reporting the empty windows as a result.

The cause is not established and is not guessed here. It is in `$OUT/srv.log` on
the worker, which needs a lease to read, and the two candidates worth separating
are an OOM under tracing (the box holds a ~68 GiB model in 128 GiB unified memory,
and this dev host OOM-killed an unrelated `agent-preflight` the same evening) and
`dgx:gpu0` itself, which went unhealthy EIGHT times on 2026-09-13. **The next
attempt should read that log FIRST**, and should print the server's tail on client
failure the way the leg function now prints the client's.

WHAT IS NOT OWED, because it is already measured: the single-window capture works
(`4e36bbae`, 1,594,621 launches, 113 MB) and produced everything in this section.
Only the second context point is missing.

### THE REFERENCE MEASUREMENT, taken at last: 13.02 tok/s at 400 tokens, a 5.08x gap (2026-09-13)

**This row has never before compared itself to sojufx on sojufx's workload.** Every
figure it quoted -- 0.257, 8.50, 12.85 -- came from a 16-token decode, while the
reference generates 400. The correction section above derives why that is not a
comparison. This is the measurement that replaces the derivation.

`dgx:gpu0` (GB10, sm_121a), `rc` job `4844371b`, source `abaa79c43`, built
`-DVLLM_CPP_CUDA_ARCHITECTURES=121a`, released UD-IQ1_S staged locally,
`--max-num-seqs 1 --device cuda --no-enable-thinking --enable-force-include-usage`.
The request body is sparkDash's own (`max_tokens=400, min_tokens=400,
ignore_eos, stop=[], temperature=0, top_p=1, stream, include_usage,
chat_template_kwargs thinking disabled`) and the rate is sparkDash's own formula,
`(completion_tokens - 1) / (tLast - tFirst)`. Prompt: "Count from 1 to 200.
Output only the numbers, separated by spaces. No other text."

| leg | tokens | tok/s | wall |
|---|---|---|---|
| `ref400_a` | 400 | **13.0215** | 31.63 s |
| `ref400_b` | 400 | **13.0176** | 31.64 s |
| `decode16`, SAME BOOT | 16 | 15.1305 | 2.19 s |

Two 400-token runs **0.030% apart**. The 16-token leg ran on the same boot
deliberately, so the comparison to this row's older numbers cannot be blamed on a
boot.

| | tok/s |
|---|---|
| sojufx sparkDash Decode Bench C1 | **66.17** |
| vllm.cpp, same workload, same formula | **13.0195** |
| **gap** | **5.08x** |

**THE DERIVATION IT REPLACES WAS CLOSE, AND IS NOW RETIRED.** The correction
section rescaled 12.85 tok/s to "~12.3 tok/s, a ~5.4x gap" by adding the QSA cost
that 400 tokens of context carries and 16 does not. Measured: 13.02 and 5.08x.
Quote the measurement, not the rescaling.

**Cumulative, all on `dgx:gpu0`: 0.257 -> 13.02 tok/s at the reference workload,
and the gap has gone 264x -> 5.08x.**

**AND THE REFERENCE'S MAIN LEVER IS NOT AVAILABLE TO US ON THIS ARTIFACT.** sojufx
runs MTP speculative decoding at K=3; the released UD-IQ1_S GGUF **carries no MTP
head**. Measured from this repository's own committed manifest of the artifact's
headers (`tests/vllm/models/qwen4_exp_gguf_manifest.inc`): 1,225 tensors, none
named `mtp`/`nextn`/`draft`/`eh_proj`/`enorm`/`hnorm`, and blocks `blk.0` through
`blk.47` with nothing higher -- exactly the 48 trunk layers and no folded 49th,
which is how llama.cpp's converter would carry one (see
`qwen3_5_gguf_weights.cpp:965-974`). The architecture defines the head and the
engine implements the method; the BYTES ARE ABSENT.

**AND NO GGUF WE COULD PRODUCE WOULD CARRY IT EITHER.** The pinned
`llama-cpp-qwen4exp` converter (PR #27742, head `6c5afc86`) is the only llama.cpp
that converts this architecture, and `conversion/qwen4exp.py:29-31` sets
`supports_mtp_export = False` / `no_mtp = True` above the comment "the MTP block
is a separate draft head; vLLM drops it too". The head's absence is the
converter's design, not a conversion oversight, so "convert one ourselves" is not
an available route without patching a second unmerged PR -- and that converter's
own `:151` records the PLE table peaking "near 300 GB of RSS", above this fleet's
largest box.

The arithmetic any plan must clear: 66.17 tok/s is a **15.1 ms** step against our
**87.6 ms** at 88% GPU-busy, i.e. **5.8x less step time**, while the two largest
kernels are 18.16 ms and 17.53 ms -- deleting both entirely leaves ~52 ms, about
19 tok/s. ~~**No combination of the identified kernel leads reaches 66 on this
artifact without speculation.**~~ **WITHDRAWN the same day** -- that sentence was
written knowing two of twenty-five kernel rows, with 41.2 ms of the step
unattributed. The full table (see "### THE FULL DECODE KERNEL TABLE, and a
retraction") puts **68.9% of kernel time in GEMM/GEMV**, splits it evenly between
cuBLAS GEMV and our own `QuantDotGemm*` kernels, and puts QSA second. A recorded
finding on this same box says every decode GEMM picks an AMPERE tactic on this
Blackwell part. Whether that is worth anything at these GEMV shapes is
unmeasured, but the closure was not earned and is retracted. This row's rule that
an apparent limit is an untraced implementation difference stands, and here it
bit the row that wrote it. See
`ISSUE-LOCAL-01M2EG6R3MRCB9ENZB9840KAXX`, which lists the three routes and says
which question to answer first.

#### THE STEP GROWS 34% WITHIN ONE 400-TOKEN RUN, AND QSA EXPLAINS A THIRD OF IT

The per-token deltas of `ref400_a` rise monotonically from **0.06630 s** at the
first token to **0.08885 s** at the last -- 15.08 tok/s falling to 11.25 tok/s,
**+22.6 ms, +34.0%** -- and `ref400_b` reproduces the shape. This is the
context-scaling the correction section argued, observed directly rather than
inferred from two runs.

**But QSA cannot account for most of it.** `|sel| = min(kv_len, block_topk * CR)`
gives 36 at the first token and 435 at the last, so against the profile's
30.5 ms at `|sel| = 1600` the gather grows **0.69 ms -> 8.30 ms = 7.6 ms**.
Measured growth is 22.6 ms. **~14.9 ms is something else that also scales with
context**, and it is now the larger unexplained term at this workload.

ONE CANDIDATE, NAMED AND NOT ASSERTED: the indexer scores EVERY complete block,
`kv_len / CR`, and that count is NOT capped by `block_topk` -- 9 blocks at
kv_len 36 against 108 at kv_len 435. The gather is capped and the scorer is not.
**DO NOT SCOPE FROM THIS PARAGRAPH.** It is a hypothesis with the right shape and
no measurement; the owed instrument is a per-kernel attribution at two context
lengths in one run, which is the same instrument
`ISSUE-LOCAL-01M2DQHP2FWXHH7GTHB17QX53Q` already owes for the `cudaFree`
population.

#### THE W9 SPEED CLAIM, MEASURED BY THE OPERATOR: at least 3.62x on the suite's own fixtures

The review noted that the operator's correctness gate (`d80f84a7`) ran pass/fail
only, so **6.52x was still an implementer report**. `rc` job `c4e4e2cb` on
`thor:gpu0` measures it instead: one clone, `test_qwen4_exp_cuda_reductions`
built and run under `nsys` at each arm, reading `QsaGatherAttentionKernel`'s own
mean from `cuda_gpu_kern_sum`.

| arm | HEAD | instances | total | mean |
|---|---|---|---|---|
| base | `6a47d4370` | 14 | 32,033,760 ns | 2.2881 ms |
| fix | `7d0d74c2c` | 17 | 8,849,920 ns | 0.5206 ms |

**THE MEANS ARE NOT COMPARABLE AND THE NAIVE 4.40x MUST NOT BE QUOTED.** The fix
arm runs THREE MORE instances than the base arm -- the new `DH` 8/16/1 case -- and
they are small shapes that drag its mean down. What the numbers support is a
bound: the 14 shared instances cost at most the whole fix total, since the three
new ones cost more than zero, so the shared mean is at most 0.6321 ms and the
speedup is **at least 3.62x**.

**3.62x IS A LOWER BOUND ON A DIFFERENT SHAPE MIX, NOT A REFUTATION OF 6.52x.**
This suite's fixtures are mostly small, and its largest runs `DH = 32`; the
implementer's 6.52x was measured at the RELEASED decode shape, `DH = 256` and
`|sel| = 2048`, where the per-row dot is 8x longer and fixed overheads are a
smaller share. Both can hold. What the operator has verified is that the kernel
got at least 3.6x cheaper on fixtures the committed suite actually runs; the
released-shape figure remains an implementer measurement, and a gate for it needs
a released-shape harness this suite does not contain.

#### What this run did NOT establish

- **The `nsys` leg FAILED, and the cause is the profiler flag rather than the
  server.** The profiled boot came up (`ready ~55s`), its first request returned
  ZERO chunks, and the next got `Connection refused`; `nsys stats` then refused
  the capture with "file does not contain StringIds table". **`nsys profile
  --duration N` TERMINATES THE PROFILED APPLICATION when the window closes** --
  the default `--kill` is a signal, not `none` -- so the 25 s window killed the
  server mid-load and left a truncated capture. The retry needs `--kill none`,
  and a window placed to open AFTER the server is serving rather than while it is
  still loading. Recorded here because the failure looks like a dead server and
  is not one. So the `cudaFree`-per-step count at the 400-token workload, and the
  GPU-busy fraction that bounds the allocator fix, are STILL OWED. The unprofiled
  legs above are unaffected -- they ran on a separate boot that served all 816
  tokens.
- **The 16-token rate moved and this run cannot say why.** It is 15.13 tok/s here
  against the 12.85 tok/s recorded at `ee0644eab`. `a69470a7f` (W8's
  device-resident k-quant MoE arm, default-ON) landed in between and is the
  obvious candidate, but **this is not an A/B** -- one boot at one SHA against a
  boot at another, weeks of commits apart in tree terms. Attributing it needs the
  interleaved same-tree harness W6 and W7 used. Recorded as an observation, not a
  result.
- W9 is NOT in this binary. `abaa79c43` predates it.

### The post-W7 profile on dgx, 2026-09-13: what W6 and W7 were worth, and what is next

**W7 on the reference box.** Interleaved same-tree A/B, BASE `3b3ed716f` (W6 only)
against FIX `ee0644eab` (W6+W7), one boot per arm, two rounds, released UD-IQ1_S
staged locally, `--max-num-seqs 1 --device cuda`, 16-token decode, median
inter-token:

| arm | round 1 | round 2 |
|---|---|---|
| BASE (W6) | -- | 8.496 tok/s, 0.11769 s |
| FIX (W6+W7) | 12.892 tok/s, 0.07751 s | 12.817 tok/s, 0.07798 s |

**0.1177 -> 0.0778 s per token, 40 ms removed, 1.51x.** The W7 scope predicted,
before the work started, "~42 ms per step removed, a step near 77 ms, and ~13
tok/s". Measured: 40 ms, 77.8 ms, 12.85 tok/s. The prediction is met, not
adjusted.

**Cumulative, all on `dgx:gpu0` with the same harness:** 0.257 -> 8.50 -> 12.85
tok/s, i.e. **50x across this campaign**, against the sojufx sparkDash C1
reference of 66.17 tok/s -- a gap of 264x at the start and **5.2x now**.

**The ranking has changed, and not toward what the pre-W6 trace predicted.**
`nsys`, 60 s window inside a 3000-token decode (**527** steady-state steps at
113.9 ms; the "~771" first written here was wrong, see the W9 correction section).
**THIS TABLE RANKS A ~1600-TOKEN CONTEXT.** `QsaGatherAttentionKernel`'s cost is
proportional to `|sel|`, so its rank does NOT transfer to the 400-token workload
the sojufx gap is quoted on without rescaling:

| kernel | share | instances | avg |
|---|---|---|---|
| `QsaGatherAttentionKernel` | **34.5%** | 6,319 | **2.545 ms** |
| cuBLAS `gemvx` (bf16, shape A) | 19.6% | 37,914 | 242 us |
| `QuantDotGemmGrouped32Kernel` | 8.8% | 25,277 | 163 us |
| cuBLAS `gemvx` (shape B) | 6.8% | 56,872 | 55 us |
| `QuantDotGemmGroupedKernel` W8 | 5.9% | 35,816 | 77 us |

`HcGroupedNormKernel` does not appear in the top twelve. Before W7 it was 40.7%
and rank 1; W7's effect is confirmed at the ranking level on the reference box,
not only end to end.

**TWO THINGS THIS OVERTURNS.** First, the next target was going to be the dense
GEMV path on the strength of the pre-W6 trace (16 ms/step) and the recorded
cuBLASLt descriptor lever. It is second, not first. `QsaGatherAttentionKernel` is
the leader at ten times the next kernel's average, and it was INVISIBLE before:
2.7% at 231 us in the pre-W6 trace, buried under the allocator. A stale ranking
would have sent the next row to the wrong kernel.

Second, and larger: **the allocator is still the top HOST cost.**

| API | share | calls | avg |
|---|---|---|---|
| `cudaFree` | **63.0%** | 52,130 | 634 us |
| `cudaMemcpyAsync` | 22.8% | 553,963 | 21.6 us |
| `cudaLaunchKernel` | 12.4% | 1,349,118 | 4.8 us |
| `cudaStreamSynchronize` | 0.5% | 198,000 | 1.3 us |

Over ~771 steps that is ~68 `cudaFree` and ~43 ms per step, at 634 us per call. **THAT STEP COUNT IS WRONG AND WAS CORRECTED THE SAME DAY (2026-09-13): it is 527, the per-step figures are ~99 calls and ~63 ms, and this whole table describes a ~1600-token context rather than the 400-token workload the gap is quoted on. See "### W9: the two numbers this row kept dividing into each other".** The paragraph below is the retracted reasoning, kept because it was published. **IT IS NOW IN DOUBT (2026-09-13).** It was derived as 60 s / 77.8 ms. The QSA attribution counts 6,319 `QsaGatherAttentionKernel` instances in the same window and the tree has 12 of 48 layers on the QSA path (`qwen4_exp.h:29-33`), which implies **527** steps, not 771 -- a 1.46x disagreement. At 527 the same totals read ~99 calls and ~63 ms per step. Neither derivation is retracted, because neither has been checked against a step counter; the CONFIDENCE is. Resolve the step count by instrumenting it before scoping anything from these per-step figures (`ISSUE-LOCAL-01M2DJ8Y4DFQMDG9GMWEK93142`).
W6 removed the per-step ADAPTER rebuild and its 378 allocations; something still
frees ~68 objects per step. The candidates are the pooled `DBuf` lifetimes in the
MoE and attention paths and any `ResidentWeight` whose `d_dev` still does not
memoise. THIS IS NOT A CLAIM ABOUT WHICH -- it is the measurement that says the
question is open, and it is the first thing the next row should resolve, because
a 634 us free is itself anomalous.

`cudaLaunchKernel` is 1,349,118 calls over **527** steps = **~2,560 per step** at
4.8 us, so **~12.3 ms per step** of launch overhead. (The "~1,750 per step at
~8.4 ms" first written here used the wrong step count; corrected 2026-09-13.) That is the number W8 exists to attack,
and it is now ~11% of the step rather than 0.5% of it.

**Owed.** A per-step attribution for the `cudaFree` population, before any row
scopes a fix for it. Do not scope from this paragraph alone.

### W9 scope: give the QSA pass-2 dot product more than one thread

Owned by `.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2DJ8Y4DFQMDG9GMWEK93142.md`,
whose ATTRIBUTED section carries the measurement this scope is built on. That
attribution is the precondition the issue itself demanded before a fix was
scoped; do not scope further QSA work from this section without re-measuring.

**What is wrong.** `QsaGatherAttentionKernel` (`src/vt/cuda/cuda_qwen4_exp_qsa.cu`)
runs two passes over the gathered KV rows. Pass 1 (`:455-478`) computes, for every
selected key `s`, the dot product `q . k_s` -- and it spreads those keys across the
block, `for (s = threadIdx.x; s < selcount; s += blockDim.x)`, because the softmax
max it feeds is associative and commutative and may be reduced in any order. Pass 2
(`:503-540`) then walks the SAME keys again in tiles of `kSelTile = 32` and
**recomputes the identical dot product inside `if (threadIdx.x == 0)`** -- on one
lane of a 256-thread block -- because the softmax denominator it feeds must be
accumulated in ascending `s`.

Measured on `thor:gpu0` (sm_110) 2026-09-13: **86.4% and 88.8%, in two independent
runs, of the whole kernel's time is those lines.** Every selected key is dotted
twice, and the second time is serial.

**Why the obvious objection does not apply.** The kernel's own comment at
`:500-501` rejects splitting pass 2 across threads: "Thread 0 walks `s` in order
and accumulates `denom` in order", and a tree reduction of `denom` would
reassociate the sum and break the bit relation to the CPU arm that this row's
goldens hold. That objection is about the SUM OVER `s`. It says nothing about the
dot product over `d` INSIDE one `s`, which is a self-contained sequential
ascending f32 accumulation whose value does not depend on which thread runs it.

**The fix.** Distribute the tile's dot products one whole dot per thread, and
leave the accumulation of `denom` exactly where it is.

For each tile of `n <= kSelTile` entries:

1. Thread `u`, for `u < n`, resolves `p` and `base` for `s = s0 + u` by the same
   arithmetic the current thread-0 loop uses, walks `d` from 0 to `DH` in
   ascending order with the same `__fmul_rn`/`__fadd_rn` pair, computes
   `w = expf(__fsub_rn(__fmul_rn(dot, scale), m))`, and writes `s_wtile[u]` and
   `s_ptile[u]`. The `base < 0` poison path keeps its current behaviour
   (`s_bad = 1; s_wtile[u] = 0.0f; s_ptile[u] = -1;`).
2. `__syncthreads()`.
3. Thread 0 alone walks `u` from 0 to `n` ASCENDING and folds `s_wtile[u]` into
   `s_denom` with `__fadd_rn` -- the same values, in the same order, into the same
   accumulator. Nothing here reassociates.
4. The value accumulation into `acc[d]` is untouched.

**The barrier count does not change, and the implementer should not add one.**
Today the tile body is: thread 0 writes `s_wtile`/`s_ptile`, `__syncthreads`, the
block reads them for the value accumulation, `__syncthreads`. After the change it
is: threads `u < n` write, `__syncthreads`, thread 0 folds `denom` (a READ of
`s_wtile`) while the block does the value accumulation (also a read),
`__syncthreads`. Two barriers before, two after. Concurrent reads of `s_wtile` are
not a hazard; the trailing barrier is what keeps the next tile's writes off them,
and it is already there.

The `reads` counter moves from thread 0 to the thread that performs the read; the
block reduction at the end of the kernel already sums per-thread counts, so the
total is unchanged. `s_bad` is already written racily by pass 1 from many threads
with the same value, so the new writers introduce no new shape.

**This is a bit-identity-preserving change, and the spec asserts it as the bar.**
Every float the kernel produces must be unchanged, bit for bit, on every fixture
the existing suites carry -- not a tolerance. That is the whole argument for
choosing this lever over the tree reduction the author already declined.

**Why it will work, stated from something already measured rather than predicted.**
Pass 1 performs exactly the same total dot-product work, with exactly the same
per-thread memory pattern (each thread walking one whole `DH`-long key row), over
exactly the same rows -- and pass 1 is inside the 11-14% remainder. The kernel
therefore already contains a measurement of the proposed access pattern at full
block width. The change makes pass 2 do what pass 1 demonstrably does cheaply.

**Prediction, written before the work starts, in the W7 style.** With `kSelTile`
unchanged at 32, the serial dot is spread over 32 threads. If pass-2 dot is 87% of
the kernel and the remainder is 13%, the kernel goes to `87/32 + 13 = 15.7%` of its
current cost, i.e. **~6.4x ON THE KERNEL**. Measure it on the kernel, at the
released shape the attribution used (`T=1, HQ=24, HKV=2, DH=256, CR=4,
block_topk=512`, `|sel| = 2048`), and record the measured value against ~6.4x
without adjusting the prediction.

**AND DO NOT TURN THAT INTO A STEP TIME BY DIVIDING IT INTO 77.8 ms.** An earlier
revision of this section did, and it was wrong. See "### W9: the two numbers this
row kept dividing into each other" below, which is the correction and is the part
to read before quoting any QSA figure anywhere.

**The step count is RESOLVED, and by arithmetic rather than by instrumenting
it** -- see the correction section below. It is 527, the window's step was 113.9 ms
and not 77.8 ms, and the 1.46x "disagreement" was two different context lengths.

Raising `kSelTile` above 32 is IN SCOPE only if it is measured, and it is bounded
by `blockDim.x` (`BlockWidthFor(DH)`), which is 256 at the released config but is
`DH` rounded up to a warp in general -- a tile wider than the block leaves entries
with no thread. `s_ptile` is a fixed `__shared__ int64_t[kSelTile]` and `s_wtile`
comes out of the dynamic allocation sized at `:610-611`; both must move together
with the constant. The default stays 32 unless a measurement on `thor:gpu0` says
otherwise.

### W9: the two numbers this row kept dividing into each other

**The defect in this row's own prose, found 2026-09-13 while dispatching W9.**
The QSA per-launch cost and the decode step time were measured on DIFFERENT
WORKLOADS, and several paragraphs above -- including the first revision of the W9
prediction and the post-W7 re-rank's per-step figures -- divided one into the
other. Every quantity below is corrected here rather than edited in place, because
the wrong readings were published and a reader who saw them needs to find the
refutation, not a silently different number.

**The two workloads.**

| | the nsys profile | the A/B harness | the sojufx reference |
|---|---|---|---|
| generated | 3,000 tokens, window `[150 s, 210 s]` | **16 tokens** | 400 tokens |
| context during measurement | ~1,300-1,900 | **~36** | ~20-420 |
| what it produced | `2.545 ms` per QSA launch, the 34.5% rank | **`77.8 ms` step, 12.85 tok/s** | 66.17 tok/s |

`QsaGatherAttentionKernel`'s cost is proportional to `|sel|`, which is
`min(kv_len, block_topk * CR)` -- 2048 at the released config. So `|sel|` is ~1600
in the profile window and **~36** in the A/B: the same kernel doing 44x less work.
`12 x 2.545 ms = 30.5 ms` is therefore a share of the PROFILE's step, and nothing
at all to do with the A/B's 77.8 ms.

**Per decode step, by context:**

| `\|sel\|` | workload | QSA per step |
|---|---|---|
| 1600 | nsys window | **30.5 ms** |
| 220 | 400-token reference, mid-run | **4.2 ms** |
| 36 | the 16-token A/B | **0.69 ms** |

**THE STEP COUNT IS RESOLVED AT 527, AND THE OTHER FIGURE IS ARITHMETICALLY
IMPOSSIBLE.** `ISSUE-LOCAL-01M2DJ8Y4DFQMDG9GMWEK93142` left it open between 527
(from 6,319 instances / 12 QSA layers) and 771 (from `60 s / 77.8 ms`), and said
to instrument it. Instrumenting it is no longer necessary:

- QSA is 34.5% of GPU kernel time and 16.079 s of the window, so total GPU kernel
  time in the window is `16.079 / 0.345 = 46.6 s`, i.e. 77.7% of the 60 s.
- At 527 steps that is **88.5 ms of GPU kernel inside a 113.9 ms step** -- 77.7%
  busy, consistent.
- At 771 steps it would be 60.5 ms of kernel inside a 77.8 ms step, which also
  reads as consistent -- **but 771 steps requires the window's step to BE 77.8 ms,
  and 88.5 ms of measured kernel cannot fit inside 77.8 ms of wall.** The
  derivation assumed the short-context step held at long context, and it does not.

So the 1.46x disagreement was never an error in either count. It is the ratio
between a 36-token context and a ~1600-token one, and the confidence that was
retracted can be restored to the 527 reading specifically.

**What this does and does not overturn.**

- **W9's fix is untouched.** Pass 2 recomputing on one lane is 86-89% of that
  kernel at any context; the head_dim-sweep attribution that established it never
  used the step time. The ~6.4x prediction is a KERNEL prediction and stands.
- **W9's end-to-end value is much smaller than the first revision claimed**, and
  it depends on context. At the 400-token reference workload QSA is ~4.2 ms of the
  step, so W9 is worth roughly 3.6 ms, a few percent -- not the ~26 ms and ~19
  tok/s that were written here. On the 16-token A/B it is under 1 ms and will not
  be visible above the noise. **A flat A/B result on that harness does not falsify
  W9**, and W9 must not be gated on one.
- **The post-W7 re-rank ranks the PROFILE's workload, not the reference's.** QSA
  is #1 at ~1600 tokens of context. At 400 tokens it is ~4.2 ms of an ~81 ms step.
  Nothing in that table transfers to the reference workload without rescaling by
  `|sel|`, and the table should not be used to pick the next row until it is
  re-measured at the workload the gap is quoted on.
- **The `cudaFree` figures inherit the same correction, in the direction that
  makes them larger per step.** At 527 steps the window's 52,130 calls are ~99 per
  step at 634 us. Those are HOST costs that do not scale with context, so they are
  ~99 per step at 400 tokens too -- against a step of ~81 ms. That makes the
  allocator, not QSA, the leading candidate at the workload this row is judged on,
  and it is what the next profile should be aimed at. **THIS IS NOT YET A CLAIM
  THAT IT IS THE COST**: `cudaFree` is host API time which can overlap device
  work, and the row still owes the per-step attribution `## Owed` already names.
  It is the reason to measure there next, not a result. **AND IT IS NOT A REASON
  TO ADD A CACHING ALLOCATOR**, which is what that sentence first invited: the
  tree already has one. `DevicePool` (`device_pool.h:111-125`) never returns a
  block to the driver, `qwen4_exp` routes its temporaries through it, and its own
  `VT_POOL_BYPASS` lane is described as reinstating "the per-op
  `cudaMalloc`/`cudaFree` sync storm this pool exists to remove". So ~99 frees a
  step is not the cost of having no cache -- it is ~99 frees ESCAPING one, which
  is a narrower and more surprising question.
  **AND THE ESCAPE ROUTE IS NAMED, BY THIS ROW'S OWN `## Owed`.**
  `vt::Qwen4ExpGatedResidual` takes a raw `cudaMalloc` per call
  (`cuda_qwen4_exp.cu:496`, freed by the `FreeGuard` at `:504-506`), and
  `qwen4_exp_forward.cpp` calls it unconditionally twice per layer (`:565`,
  `:638`, both inside the loop at `:491`) plus once after (`:747`). At `L = 48`
  that is **97 per step against 98.92 measured, a 1.9% residual** -- and at
  634 us per free, **61.5 ms of the ~63 ms**. The same trace corroborates it from
  the other side: `cudaMalloc` ran **52,143** times against `cudaFree`'s
  **52,130** -- a strictly paired population differing by 13 over 60 s -- and the
  whole cost is on the free: **633.69 us against 4.04 us, 157x**, because
  `cudaFree` synchronises and `cudaMalloc` does not. **AND THAT NAMES A SMALLER
  FIX THAN `## Owed` DOES:** the same trace has `cudaFreeAsync` at **1.30 us**
  over 50,553 calls, so the stream-ordered pair is already carrying a comparable
  population in this run at negligible cost, with no signature change and no
  re-entrancy argument -- subject to TWO conditions, both already measured
  elsewhere: the graph-capture interaction `cuda_mla_attn.cu:454` records, which
  this row cares about because W8 exists to make capture possible; and
  `ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ` (`abaa79c43`), which measured
  `DevicePool`'s uncapped retention driving `MemAvailable` from 58.2 to 13.8 GiB
  in two minutes on GB10. **THOSE TWO FINDINGS PULL OPPOSITE WAYS** -- this one
  says ~97 allocations a step should be in a pool and are not, that one says the
  pool already retains more than the box can afford -- and the driver's
  stream-ordered pool retains by default too, so "put it in a pool" is not on its
  own an answer. THIS IS AN ARITHMETIC MATCH AND NOT A MEASUREMENT;
  instrumenting the three call sites settles it in one run. See
  `ISSUE-LOCAL-01M2DQHP2FWXHH7GTHB17QX53Q`, which also refutes a per-step `Drain`
  as the source.

**MEASURED 2026-09-13, AND THIS PARAGRAPH'S RESCALING IS RETIRED BY IT:** at the
reference's own 400-token workload we run **13.0195 tok/s**, a **5.08x** gap, two
runs 0.030% apart on `dgx:gpu0`. See "### THE REFERENCE MEASUREMENT, taken at
last". The reasoning below is kept because it is why the measurement was taken;
its ~12.3 tok/s and ~5.4x are a derivation and must not be quoted as a result.

**AND THE COMPARISON TO sojufx IS ITSELF ON THE WRONG WORKLOAD.** The reference
generates **400** tokens; this row's 12.85 tok/s was measured generating **16**.
The two are not the same benchmark, and "5.2x" is a ratio between them. Rescaled to
the reference's workload -- adding the ~4.2 ms of QSA that 400 tokens of context
costs and that 16 tokens does not -- the step is ~81 ms and the rate is **~12.3
tok/s, a ~5.4x gap**. That is a small correction to the number and a large one to
the method: this row cannot keep quoting a 16-token rate against a 400-token
reference. **THE 400-TOKEN RATE IS OWED AS A MEASUREMENT** and the rescaling above
is a derivation, not a result. The harness already accepts the token count; the
next dgx window should spend part of itself on `max_tokens=400` with the
reference's own prompt, so the headline ratio is finally one benchmark against the
same benchmark.

**The rule this row now carries.** Never quote a per-launch kernel cost and a step
time from different runs in the same sentence. Never quote a rate against a
reference that generated a different number of tokens. For an attention kernel whose work
is proportional to context, the workload is part of the measurement, and a profile
taken at 1,600 tokens does not describe a benchmark run at 400.

### W9 OUTCOME, as landed at `054a8910c` (operator, 2026-09-13)

**What landed.** Pass 2's tile evaluates one whole dot product per thread; only
the sum over `s` stays on thread 0, so no reduction was reassociated. Four
commits: the kernel change, the fold gate the review demanded, and two records.

**The operator's gate, rerun on the final head `af417e4ee` rather than on the
implementer's report** (`rc` job on `thor:gpu0`, sm_110, fresh clone, selected
case count asserted for every suite):

| suite | cases | assertions |
|---|---|---|
| `test_qwen4_exp_cuda_reductions` | 20/20 | **19,678** |
| `test_qwen4_exp_qsa` | 14/14 | 7,263 |
| `test_qwen4_exp_qsa_device` | 12/12 | 4,697 |
| `test_qwen4_exp_cuda` | 12/12 | 351 |
| `test_qwen4_exp_qsa_block` | **12/13** | 7,438/7,439 |

The assertion count on the first suite went 213 -> 19,678, and that IS the
repair: the fold gate now exercises the property W9 rests on. The fifth suite's
single failure predates W9 and is measured to (`ISSUE-LOCAL-01M2E18YDQ8M5Y5P5SPFXD1D99`).

**Speed: at least 3.62x on the kernel, measured by the operator.** See the
subsection above; the naive 4.40x mean ratio is invalid because the fix arm runs
three more instances. The implementer's 6.52x at the released `DH = 256`,
`|sel| = 2048` shape is not contradicted and is not gated.

**THE ROCm ARM IS COMPILED BY NOTHING, AND THAT WAS FOUND HERE.** W9's repair
needed the probe field in `include/vt/ops.h` and handling in all three arms, so it
touched `src/vt/rocm/rocm_qwen4_exp_qsa.hip`. `check-tree-compiles` reports
`src/vt/cuda/cuda_qwen4_exp_qsa.cu` as "in scope that no target in this
configuration compiles" and does not see the `.hip` at all, and the operator's
gate is CUDA. So a typo in that file would have reached `main` unbuilt.
`hipcc -fsyntax-only --offload-arch=gfx1151` on `strix:gpu0`, both arms, `rc=0`
each, with different `.hip` sha256 proving the file really differs. **This is a
standing gap, not a W9 one**: any change touching a `.hip` has the same exposure.

**What the review changed, and why the FAIL was right.** The code was correct
throughout; what failed was that its central claim was enforced by nothing. Two
reassociations -- a descending fold, and the warp-shuffle tree this kernel's own
header declines -- passed all three committed suites while demonstrably moving the
output. Merging on "the code is correct" would have left the declined lever free
to be added later with every gate green.

### Out of scope for W9

The narrow decode grid (24 blocks on 48 SMs) recorded in the issue's structural
paragraph. The attribution REFUTED it as the dominant cost, and widening the grid
is a separate change that cannot be justified from these numbers.

### W9 gates

1. `test_qwen4_exp_cuda_reductions` and the QSA device suites, green on
   `thor:gpu0`. A doctest filter that selects zero cases exits 0; assert the
   selected case count, never the exit code alone.
2. **Bit identity, proven as a difference and not asserted as prose, and NO
   EXISTING CASE GATES IT.** `test_qwen4_exp_cuda_reductions.cpp:37-46` holds
   `Qwen4ExpQsaGatherAttention`'s CUDA arm to its CPU arm by a DERIVED BOUND, not
   by equality, because the two spell `exp` differently. That bound is wide
   enough to swallow a reassociated denominator, so a green run of that suite is
   NOT evidence for this change -- it is the trap this gate exists to avoid. The
   required comparison is CUDA-BEFORE against CUDA-AFTER: capture the gather's
   full output for every QSA fixture at BASE `7a9020304`, capture it again at FIX
   with the same binary configuration, and require the bytes to compare equal.
   (`Qwen4ExpQsaCompress` is untouched by this change and its existing byte
   equality stays green; do not confuse the two ops' bounds.)
3. Red-first: the smallest test that fails for the intended reason, captured red
   before the change.
4. `compute-sanitizer --tool racecheck` over the QSA suite, 0 hazards, with the
   control that a deliberately deleted `__syncthreads()` between step 1 and step 3
   above reports hazards. A racecheck run with no positive control proves nothing
   about the new barrier.
5. `scripts/agent-preflight.sh` green.

### W9 evidence required

- The red capture, the focused green, and the full gate.
- The bit-identity comparison, as the two captured byte streams and the compare.
- The racecheck run AND its positive control.
- **The kernel-level before/after IS the gate for this row's speed claim**, not
  interim evidence. Measure `QsaGatherAttentionKernel` itself on `thor:gpu0` at the
  released shape the attribution used, and report it against ~6.4x.
- **The 16-token A/B harness the W6 and W7 rows used DOES NOT APPLY HERE**, and
  running it would be the mistake the correction section names: at `|sel| = 36`
  this change moves under 1 ms of a 77.8 ms step. A flat result on that harness is
  not a W9 result in either direction. Do not gate W9 on it and do not quote it.
- An end-to-end number is owed at LONG CONTEXT -- the workload where the 2.545 ms
  was measured -- on `dgx:gpu0` when the box is healthy. `dgx:gpu0` has been
  unhealthy repeatedly this session; `thor:gpu0` (build `sm_110`, never `121a`) is
  the development target.
- A re-rank `nsys` after it lands. The last two rows each overturned what was
  planned next; assume this one does too.

### W9 stop conditions

- Any float changes. The bar is bit identity; a drift means the change is not the
  one this spec scoped, and the finding is a stop, not a tolerance to widen.
- Racecheck reports a hazard the positive control does not explain.
- The measured speedup is below 2x on the kernel. That would falsify the
  attribution, not the fix, and the attribution is what would then need redoing.

### W8 scope: a device-resident arm for the k-quant MoE block

Owned by `.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2AAACGC0SXYXFN2JQ3GFEAZ.md`.
Scoped from the dgx trace that ranked W6 and W7; re-rank against the CURRENT step
before accepting any number here.

**What is wrong.** `MoeBlock` (`qwen3_5.cpp:7178`) has three arms: NVFP4 fused,
bf16 fused, and a "reference path" for everything else. A GGUF k-quant
checkpoint has neither fp4 nor bf16 experts, so EVERY GGUF MoE model this tree
serves takes the third. That arm copies the hidden state to host and
synchronizes once per layer, downloads the router top-k, and makes three
`KqGrouped` calls (`:6133-6145`) that each upload an activation, launch ONE
grouped GEMM and `Download` an f32 result -- a blocking drain -- with the SwiGLU
in a host loop between them, and the routed output assembled in a host
`std::vector` before being uploaded again for the combine.

**Why it is worth doing now rather than earlier.** When the decode step was
3.95 s this was 1 ms of it. The trace measured `cudaStreamSynchronize` at 367
calls and 0.001 s per step against 2.25 s of `cudaMalloc`, which is why W6 came
first and why this issue was explicitly recorded as a CAPTURABILITY defect
rather than a speed one. W6 removed the allocator and W7 removed the norm, so
the step is now dominated by what is left, and the same trace counted **2,534
`cudaLaunchKernel` per step at ~20 ms** -- 0.5% of a 3.95 s step and a much
larger share of a ~100 ms one. The tree states the consequence at `:7191-7193`:
the fused arms are "capturable", the reference path is "not the capture target".
While the MoE block leaves and re-enters the device every layer, CUDA graph
capture is impossible for the whole model.

**No new kernels are required, and that is what makes this tractable.**
`vt::MatmulBTQuantGrouped` already runs the three grouped GEMMs on device for the
encodings the shipped artifacts use (`IsCuda32BlockKeepQuantSupported` covers
IQ4_NL / Q5_0 / Q4_0; `IsCudaKeepQuantSupported` covers the Q8_K family).
`vt::OpId::kMoeSiluMul` is registered on CUDA (`cuda_moe.cu:941`) with the
signature this needs, `(Queue&, Tensor& out, const Tensor& a, const Tensor& b)`.
`MoeRouterTopK` and the combine are already device ops. What is missing is an arm
that keeps the intermediates on device BETWEEN them.

**Design, and it is the bf16 arm's shape rather than a new invention.** Add an
arm selected exactly where `MoeBlockBf16Cuda` is, under the same shape of
condition that arm uses: the required ops registered, and a default-ON
environment gate (`VT_MOE_KQ_FAST`, with `=0` restoring the reference loop for a
same-binary A/B and as the correctness oracle). That gate is the precedent at
`MoeBf16FastEnabled()` (`:909`) and it is what makes this reviewable: the
reference path stays, and the new arm is compared against it rather than
replacing it unobserved.

**WHAT LANDED DIFFERS FROM THE PARAGRAPH ABOVE IN TWO WAYS, AND THIS SECTION
RECORDS THE LANDED SHAPE RATHER THAN THE SCOPED ONE.** The commit argues each
deviation; the spec is corrected here so that a reader is not told a shape was
built that was not.

*The name is `MoeBlockKqDevice`, not `MoeBlockKqCuda`.* The scoped name says
CUDA and the arm does not: it asks `vt::OpRegistered(..., d.q.device.type)` for
each op it calls, which is what `check-device-leakage.py` requires of this
device-agnostic layer and what the bf16 arm above it already does. A `Cuda`
suffix on a predicate that never names CUDA would be the wrong name for the
code. The consequence is that the arm is live on every backend the three ops are
registered on — CPU, CUDA, ROCm and Tenstorrent — which is why the suite gates
three of those four and why the fourth is filed
(`.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2D7X94RTPJ453QGHYKQZPY8.md`).

*There are SEVEN conditions, not three, and none of them is a device term.*
Counted at `qwen3_5.cpp:7411-7415`, the predicate is NINE `&&`-joined terms:
`!fp4`, `!w.expert_gate_kq.Empty()`, `T == 1`, three `vt::OpRegistered` calls
(`kMatmulBTQuantGrouped`, `kMoeSiluMul`, `kCastBf16`), `Qwen35GroupedMoeEnabled()`,
`MoeSelFpCalls() == 0` and `MoeKqFastEnabled()` — seven conditions if the three
op-table questions are read as the one condition "the arm's ops are registered
here", which is how the prose below groups them. An earlier version of this
paragraph said FIVE and that was simply a miscount. Two of the extra ones are the
arm's own preconditions: a non-empty keep-quant tower set, and `T == 1`, which is
the broadcast's scope (below). The other two refuse an INVISIBLE FALLBACK rather
than a defect. `Qwen35GroupedMoeEnabled()` is the
existing grouped-vs-per-expert route switch and is already false when expert
streaming was asked for, so taking this arm anyway would silently do neither.
`MoeSelFpCalls() == 0` is the `VT_MOE_SEL_FP` tap, which reads host buffers this
arm never materialises — an operator who armed the tap would otherwise get the
fast arm and a silently empty tap. That term is load-bearing and it is gated by
its own ctest registration (`test_qwen35_moe_kq_device_sel_fp`), because the tap
count is cached in a process-static and cannot be armed from inside a case.

The broadcast the grouped kernel already implements (`Pa == 1 && P > 1`,
`cuda_quant_dot.cu`) means the routed activation need not be gathered on host at
all when `T == 1`: the hidden is one row and every routed pair reads it.

**The bar this must clear, stated before the work starts.** PER-PAIR BIT
IDENTITY with the reference arm, not a tolerance. The bf16 arm's own comment
claims exactly that of itself ("the fused output is per-pair bit-identical to
it"), and the k-quant path has the same property available because both arms
reach the same `kMatmulBTQuant` core with the same `eids` slice. A tolerance
here would hide a routing or ordering defect, which is the class of bug this
seam has produced before (#2249 item 4, the rank-3 tower that "matched" by
shape).

**THE LANDED BAR IS THAT, EXCEPT AT THE SwiGLU, WHERE IT IS ONE bf16 ULP.** The
reference computes `Silu()` with `std::exp` and the device op computes it with
`expf`, so identity is not this arm's to claim. It is also not something the
block comparison can measure: a 1-f32-ULP move in `silu(g)` changes
`bf16(silu(g)*u)` in 1.44e-5 of N(0,3) samples (288 of 20,000,000, measured), so
the block fixture's 96 SwiGLU elements would pass with probability ~0.9986
against a kernel that disagreed on every input. The step is therefore gated AT
THE OP, over 2^22 pairs per backend, against the reference arm's own host
formula, with the observed disagreement count printed on the green run. The CPU
sweep must be EXACTLY equal, because `cpu_ops.cpp:733` is that same formula.

*The device `expf` IS THE ORACLE'S OWN SPELLING, and that — not a local
precedent — is why this arm is default-ON.* An earlier version of this section
justified the deviation by citing `cuda_moe.cu:823-827`, which calls it an
ACCEPTED deviation. That is a fact about THIS tree and it is the wrong argument:
a default-ON, token-visible change cannot rest on our own precedent. The oracle
settles it. vLLM's `silu_kernel` is
`return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));`
(`csrc/libtorch_stable/activation_kernels.cu:158`, anchored in this tree at
`.agents/specs/vt-act-round-polarity.md:90`), and `expf` there is the CUDA
device `expf` — the same function `vt::MoeSiluMul` calls
(`src/vt/rocm/rocm_moe_router.hip:24`, `Silu(float x) { return x / (1.0f +
expf(-x)); }`, whose own comment quotes that upstream line at `:44-45`). The
host `std::exp` in the reference arm is OUR artifact, introduced by a host loop
vLLM does not have. So this arm does not move AWAY from the oracle at the
SwiGLU; it moves TOWARD it, and the one-ULP band is the distance between our
host reference and the oracle rather than the distance between this arm and
correctness.

*The bf16 precedent is default-ON on a WEAKER claim than this one.*
`MoeBlockBf16Cuda` ships default-ON behind `MoeBf16FastEnabled()`
(`qwen3_5.cpp:909-915`) and its header states only that each output row "stays
within the near-tie band of the reference path it replaces"
(`qwen3_5.cpp:6883-6884`) — a band it measured nothing against. This arm makes
the same trade with a measured bound and an oracle anchor, so default-ON here is
strictly better supported than a default that already ships.

**Tests.** A red-first case that fails for the intended reason: run one MoE
block through both arms on the same inputs and assert byte equality of the
routed output, with the new arm forced on and off by the gate. The red must come
from the ARM SELECTION, not from a numerical bound -- a test that only checks
tokens cannot see this, exactly as it could not see W6.

**What landed is `tests/vllm/models/test_qwen35_moe_kq_device.cpp` plus four
ctest registrations of it:** the default one, `..._ref` with `VT_MOE_KQ_FAST=0`,
`..._sel_fp` with `VT_MOE_SEL_FP=4096`, and `test_rocm_qwen35_moe_kq_device`,
which runs the same binary filtered to its ROCm cases. Arm selection is read
through the `vllm::MoeKqDeviceCalls()` probe on CPU, CUDA and ROCm; the SwiGLU
sweep above is in the same file on the same three; Tenstorrent is unreached and
filed.

*The fourth registration exists because the documented ROCm gate is a NAME
filter.* `docs/ROCM.md:24` runs `ctest --test-dir build-hip -R
'rocm|cross_device'`, and none of the first three names carries either token, so
the ROCm cases never ran in the lane a ROCm operator is told to run — while the
arm is DEFAULT-ON on ROCm and `strix:gpu0` (gfx1151) is a live fleet device. A
doctest filter that selects nothing exits 0, so that registration also carries
`FAIL_REGULAR_EXPRESSION "test cases: +0 [|]"`, which reddens a zero-selection
run rather than letting a renamed case turn it into a green that measured
nothing.

*The SwiGLU sweep asserts a disagreement RATE as well as a magnitude.*
`worst_ulp <= 1` bounds how far a disagreement goes and says nothing about how
many elements disagree. MEASURED on `thor:gpu0`: with `MoeSiluMulKernel`
multiplied by `1.001953125f` (1 + 2^-9, half a bf16 ULP) the CUDA sweep reports
`disagreements=1510125 (3.600e-01) worst=1 bf16 ulp`, and the ULP bound PASSES
while 36.0% of the op's outputs have moved. `kSweepMaxDiff` is therefore
`kSweepN / 1024` (4,096 elements, 9.77e-04): ~455x above the measured 2.146e-06
so that an unmeasured backend's `expf` has room, and ~369x below that 36.0%.
Restored byte-for-byte afterwards (`cuda_moe.cu` back to
`95930fec20051876525e951ade1b93e758f951f8357be608fa19b99ed04db2ba`), with the
binary proved changed either side (`c766c68e…` mutant vs `16d1d4e3…` restored).

**WHAT THE PER-ELEMENT RATE MEANS ON THE ARTIFACT THIS ARM SERVES.** The
conversion above is onto the 96-element block fixture, which is the reassuring
direction and not the production one. Qwen3.8-Flash-Next is 48 layers with a MoE
block on every one, routed at top-10 into experts of `moe_intermediate_size =
640` (this spec, `:18-20`, `:373`, `:1042`), so ONE decode token pushes
`48 x 10 x 640 = 307,200` elements through `vt::MoeSiluMul`. At the measured
2.146e-06 that is **0.66 perturbed elements per decode token**, each up to one
bf16 ULP before the down GEMM's q8_0 activation quantizer; treating the sites as
independent, about 48% of tokens carry at least one. This is a real, routine,
per-token deviation from the reference arm and not a rounding curiosity — which
is exactly why the sweep, and not the block fixture, is the instrument.

**NOTHING IN THIS ROW GATES TOKENS.** The Gate below is ctest suites only; the
arm has no token-exact run, on any artifact, on either arm of `VT_MOE_KQ_FAST`.
The per-token figure above says what such a run would be measuring, and the row
does not claim one. It is owed by
`.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2AAACGC0SXYXFN2JQ3GFEAZ.md`
alongside the `nsys` A/B.

**Gate.** `test_qwen4_exp_moe`, `test_qwen4_exp_moe_sel_fp` (the pair that pins
the borrow identity, per the W6 review's M5), plus the qwen3_5 MoE suites, since
`MoeBlock` is shared. The CUDA arm needs a lease; `test_qwen4_exp_hc_device` is
NOT a device gate despite its name. These are ctest suites; see the paragraph
above for what is NOT here.

**Owed evidence.** An `nsys` A/B on `dgx:gpu0`, arms interleaved, reporting the
per-step `cudaLaunchKernel` count and `cudaStreamSynchronize` count on both
arms, not only tok/s. The launch count is the number that says whether graph
capture became possible; tok/s alone would not distinguish this change from a
kernel that merely got faster.

**Out of scope.** CUDA graph capture itself -- this change makes it POSSIBLE and
a separate row does it. The NVFP4 and bf16 arms. `KqResidentSlice` and the
per-expert path, which stays as the `VT_QWEN35_GROUPED_MOE=0` fallback.

### W7 scope: make the hyper-connection grouped norm fp32 and parallel

Owned by `.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2C8HBDD9VG6AJMPM9PTN80S.md`.
Scoped AFTER W6 landed, against the 116.9 ms step W6 produced, never against the
3.95 s one it retired.

**Why this one next.** `nsys` ranks it: `HcGroupedNormKernel` is 40.7% of all GPU
kernel time, 96 instances per decode step at ~435 us each, ~42 ms of a 101 ms
kernel budget. With W6 landed the step is 116.9 ms, so this kernel is ~36% of a
decode token.

**The two defects, ANCHORED AT THE PARENT `49ffc61cf` because the fix has since
landed and moved every line below.** `cuda_qwen4_exp.cu:258` launched ONE THREAD
PER (token, hc stream), `GridFor(T * hc)` at `:402`. At `hc_count = 4` and decode `T = 1` that
is FOUR threads on 48 SMs, each walking `H = 2560` twice, and the sum of squares
accumulated in `double` (`:271-275`). Precision and parallelism are separable and
both are in scope.

**This is not a parity decision, and the spec says so rather than leaving it to
the implementer.** `qwen4_exp_hc.h:99-104` already states the rule: the double is
a HOST REFERENCE convention, "upstream runs the norm in fp32 and vLLM likewise",
and "THE DEVICE ARM IS THE THING THAT MUST BE FP32-ACCUMULATE". The device arm is
therefore the outlier, and moving it to fp32 moves it TOWARD the oracle.

**The existing gates already admit the change, which is what makes it cheap.**
The device arm is not held bit-exact against the CPU arm.
`test_qwen4_exp_hc_device.cpp:76` gates the golden widths at `kTol = 1e-5`
absolute; its model-width case at `:378` gates a RELATIVE `4e-5` (`:442`),
documented
there as "6.6x the sqrt(K)*u random-walk bound for K = 10240" -- a bound derived
for FP32 unit roundoff, beside the sentence "because torch runs this in fp32
too". A serial fp32 walk is ~sqrt(K)*u ~= 6e-6 relative; a block tree reduction
is BETTER at ~sqrt(log K)*u. Both sit inside a bound this tree derived before W7
existed. Do NOT widen either tolerance. If the change cannot meet them, that is a
finding about the change, not about the bound.

**WHICH FILE IS THE CUDA GATE, because an earlier draft of this section named the
wrong one.** `test_qwen4_exp_hc_device.cpp` is the CPU ARMS' gate against the
transformers goldens; its own head says "Nothing below runs on a device", so it
executes none of this kernel and cannot hold it. The CUDA gate is
`tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp`. MEASURED on `thor:gpu0`
(sm_110, CUDA 13.0.88): corrupting the `1 +` gamma fold in `HcGroupedNormKernel`
reddens `test_qwen4_exp_cuda_reductions` at 7 of 18 cases, worst `max|diff|`
0.868741 against its 1.95703e-06 bound (the grid-cap case), while
`test_qwen4_exp_hc_device` stays SUCCESS at 11/11 cases over 516 assertions. The tolerances above still stand as the numerical
contract; the file that enforces them on the device arm is the CUDA one. Run
`test_qwen4_exp_hc_device` too -- as the CPU-arm gate that must not regress, not
as evidence about the kernel.

**Design.** One block per (token, hc) group; a block-wide reduction for the sum
of squares in fp32; the normalize-and-write loop that follows is
order-independent and parallelises across the block with no numerical question.
The `1 +` gamma fold stays f32 and stays where it is (`49ffc61cf:288-290`, now
`:357-363`) -- it is
upstream's `1.0 + self.weight.float()` and #2218 records what dropping it looks
like. `eps` stays INSIDE the rsqrt, added to the mean square, with the narrowing
point unchanged.

**Tests.** The red must come from the SHAPE rather than from the width: a case
that fails because the kernel computed one group and broadcast it, or walked the
wrong stride, is the one that discriminates a bad parallelisation. The CUDA cases
live in `test_qwen4_exp_cuda_reductions.cpp` under the `CUDA W7:` names, and
`test_qwen4_exp_hc_device.cpp` must stay green unmodified as the CPU arms' gate.

**NO RED-FIRST WAS AVAILABLE, and this paragraph records why rather than letting
the sentence above read as an unmet obligation.** The W7 fixture cannot be made
to fail on the pre-change kernel. Both arms walk the group ASCENDING, the
pre-change device arm accumulates in `double`, and at the fixture's synthetic
inputs the pre-change kernel reproduces the CPU reference EXACTLY: an independent
rebuild of the pre-change kernel against the new fixture measured `max|diff| = 0`
on every case. A red-first case would therefore have to fail for the width, which
is the one thing this change is allowed to move, and a case that fails on the
correct kernel is not a gate.

The discriminating evidence is a MUTATION BATTERY plus an in-gate separation
probe instead. Four of the five kernel mutations redden
`test_qwen4_exp_cuda_reductions`, and two of those four -- dropping the group
grid stride and collapsing the cross-warp fold -- are invisible to every
committed golden case, which is the property the new fixture exists for. The
fifth (a trailing `__syncthreads()` on the group loop) is an EQUIVALENT MUTANT:
the loop's two remaining barriers already order both cross-iteration shared
accesses on a block-uniform trip count, `compute-sanitizer --tool racecheck` on
the grid-cap case reports `0 hazards displayed` on the committed bytes, and
adding the barrier back gives digit-for-digit identical output. The separation
probe runs inside the gate and measures ONE defect, which is not a member of
that battery and is not its floor either. It replays the model-width case with
every group of token 0 forced to group 0's data, which is what a kernel that
computed one group and broadcast the reciprocal `r` would produce
(`test_qwen4_exp_cuda_reductions.cpp:1754-1783`), and it reports
`ratio = 252555.5x` between THAT defect's signal and the case tolerance. So the
discrimination band for a broadcast `r` is measured rather than asserted. It
bounds no other defect, and nothing here makes it the smallest signal of the
five mutations.

**Owed evidence.** An `nsys` re-measurement on `dgx:gpu0`, same harness as the W6
A/B (interleaved arms, one boot per arm, released UD-IQ1_S staged locally), giving
the new per-step kernel share for this kernel and the new tok/s. The prediction to
test, stated so it can be wrong: ~42 ms per step removed, a step near 77 ms, and
~13 tok/s. Record what is measured, not this number.

**Out of scope.** The CPU reference's double (it is the oracle these gates use),
`vt::RmsNormGroup`, the PLE block's norms, and the cuBLAS `gemvx` path -- the
latter is the NEXT item at 15.9% and gets its own row.

### W7 measured on thor (sm_110), 2026-09-13: ~1.5x, and dgx came in later the same day

**Harness.** An interleaved same-tree A/B on `thor:gpu0` (NVIDIA Thor, sm_110),
one boot per arm, two rounds alternating BASE and FIX, the released
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S staged to local disk, the server at
`--max-num-seqs 1 --device cuda` with no other engine flag, a 16-token decode,
and the median inter-token interval as the statistic.

| arm | commit | round 1 tok/s | round 1 s/token | round 2 tok/s | round 2 s/token |
|---|---|---|---|---|---|
| BASE, the W7 parent | `49ffc61cf` | 4.6564 | 0.21372 | 4.9773 | 0.21430 |
| FIX | `ac04275b8` | 7.3755 | 0.14542 | 7.2675 | 0.14629 |

The median per-token interval goes 0.2140 s -> 0.1459 s. That is ~68 ms removed
per token, about 1.5x. Peak resident memory is unchanged: `VmHWM` 77,361,940 kB
on BASE against 77,355,492 kB on FIX. No arm printed `out of memory`,
`bad_alloc`, or a CUDA error.

**THE BYTES MEASURED ARE THE BYTES THAT LAND.** The A/B ran `ac04275b8`, which is
an earlier revision of this same commit; the revision that carries this record is
`d7e0e9cf2`. `git diff ac04275b8 d7e0e9cf2 -- src/` is EMPTY, and the only
`tests/` difference is one comment block in
`test_qwen4_exp_cuda_reductions.cpp`. Every later revision was prose and records.
The rebase onto `43622bc37` then reproduced the commit's patch byte-for-byte:
`git diff d7e0e9cf2^ d7e0e9cf2` and `git diff HEAD^ HEAD` are identical files.
A measurement of `ac04275b8` is therefore a measurement of the executable bytes
this row lands.

**THIS WAS THOR, NOT DGX, AND THE DGX MEASUREMENT WAS OWED WHEN THIS WAS
WRITTEN. IT WAS TAKEN LATER THE SAME DAY (`f97e8451a`) and the paragraphs below
are kept as written rather than edited, because they were published.** On
`dgx:gpu0`: 0.1177 -> 0.0778 s per token, 40 ms removed, **1.51x**, 8.50 -> 12.85
tok/s. The prediction was met on all three axes. The owning issue
(`ISSUE-LOCAL-01M2C8HBDD9VG6AJMPM9PTN80S`) is CLOSED (`685b856e0`). Read the
"owed" and "OPEN" statements below as the state on the morning of 2026-09-13, not
as the state of this row.

Every other
number in this row -- the 116.9 ms step, the 8.57 tok/s, the `nsys` kernel
shares -- comes from `dgx:gpu0`, a GB10 at sm_121a. This one comes from
`thor:gpu0` at sm_110, which is a slower box with a larger step. `dgx:gpu0` read
`unhealthy (no contact)` across three separate outages on 2026-09-13, so the
re-measurement against the 116.9 ms step could not be taken. Do NOT read the
thor figure against the sojufx reference or against the W6 dgx numbers; they are
different machines and the comparison is not defined.

**THE 42 ms ATTRIBUTION IS NOT CONFIRMED BY THIS.** The `nsys` attribution that
motivated W7 -- `HcGroupedNormKernel` at 40.7% of GPU kernel time, ~42 ms per
step -- was measured on dgx. The 68 ms per token removed here was measured on
thor, on a larger step. The direction and the rough magnitude agree. That is all
they do. Nothing here confirms the 42 ms figure.

**THE PREDICTION IS NOT RETIRED.** The W7 scope section above states the
prediction so it can be wrong: ~42 ms per step removed, a step near 77 ms, and
~13 tok/s. It was written for dgx and it has NOT been tested on dgx. On thor the
step went 0.2140 s -> 0.1459 s. The prediction stands OWED a dgx measurement,
and the owning issue stays OPEN for it.

> **SUPERSEDED THE SAME DAY.** The dgx measurement was taken at `f97e8451a`:
> 40 ms removed, a 77.8 ms step, 12.85 tok/s, against a prediction of ~42 ms,
> ~77 ms and ~13 tok/s. The prediction is retired as MET, and the owning issue is
> CLOSED. The paragraph above is the state before that run and is kept for
> provenance.

### W6 scope: hoist the MoE adapter onto the model

Owned by `.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS.md`.

**Change.** `Qwen4ExpMoeBlockWeights` is composed inside the layer loop
(`qwen4_exp_forward.cpp:673`). Build it ONCE per layer and hold it on the layer's
weights, exactly as the GDN twin does at `:587` -- `if (!lw.gdn_block.has_value())
lw.gdn_block.emplace(Qwen4ExpGdnBlockWeights(lw.gdn, p));`. Add the parallel
`std::optional<MoeBlockWeights>` field beside `gdn_block` on
`Qwen4ExpLayerWeights` and emplace it the same way. The seam, the config
projection and `RunQwen4ExpMoeBlock` do not change.

**Why it works.** `ResidentWeight` writes its residency memo (`d_dev`, and the
host-alias decision) onto the handle it is given.
`BorrowWholeOwnedTensor` carries every field EXCEPT `d_dev`, so a per-step
adapter presents a null memo every step and every tower takes the
`cudaMalloc`+copy arm again. A handle that outlives the step keeps the memo.

**Three constraints the existing comments already state, and a reviewer must
check each.** (1) `Qwen4ExpMoeBlockWeights` takes a NON-CONST reference and
mutates it through `OwnedBytes::KeepAlive()`, so it is not a pure function of the
layer and must not be re-evaluated per placement arm -- the existing code already
builds it outside the placed body for this reason. (2) The pass-through must stay
a BORROW. When the GDN adapter spelled the same thing as assignment,
`OwnedTensor`'s implicit copy DEEP-COPIED ~115 MiB per linear layer per step and
`ResidentWeight` then handed cuBLASLt a pointer into a buffer destroyed at the
end of the layer -- `compute-sanitizer` reported it as `Warp illegal address`
with every declared extent correct (#2476). (3) The hoisted handle must outlive
every queued kernel that reads it, which holding it on the model satisfies and a
function-local does not.

**Reachability.** The call site is the production decode path
(`ForwardQwen4ExpForConditionalGeneration` -> the layer loop), so the mutation
the fresh reviewer owes is deleting the hoist's call site in a scratch copy and
confirming the focused gate goes red.

**Tests.** A red-first test that fails for the intended reason. The defect is
"residency is re-established per step", so the test must observe the SECOND
forward, not the first: assert that a second decode step over the same layer
performs no new device staging. `vllm::load_stats::Snapshot().device_upload_bytes`
is the counter that expresses it and `load_stats::Reset()` exists for tests.
A test that only checks the output tokens CANNOT see this defect -- the tokens
were always correct, which is why twelve waves of green gates never caught it.

**Gate.** The row's C++ suite, plus the focused MoE and qwen4_exp tests. The
speed claim is NOT a gate and must not be asserted from a unit test: it is
re-measured on `dgx:gpu0` with the same harness as the profile above, by the
operator, and recorded in `## Outcome`.

**Out of scope.** `BorrowWholeOwnedTensor` itself. Carrying `d_dev` through the
borrow would fix every caller at once and is the tempting change, but the memo
records a DEVICE allocation whose lifetime is tied to the source handle, and
copying that pointer into a second handle that can outlive it is how #2476's
use-after-free was built. If a sweep of other per-step adapters is wanted it is
its own row, with its own review.

**Device residency changes, and that is the mechanism rather than a side
effect.** The `d_dev` deleter that `ResidentWeight` installs
(`include/vllm/model_executor/models/dense_attn_block.h:243`,
`w.d_dev = std::shared_ptr<void>(p, [bk] (void* q) { bk->Free(q); })`) hangs off
whichever handle was staged. Today that handle is the per-layer temporary, so
the deleter runs at the end of each layer -- those are the 378 `cudaFree` calls
per step the profile above measured. After the hoist the deleter hangs off a
model-held handle, so every expert tower stays device-resident for the life of
the model. Steady-state DEVICE footprint therefore moves from roughly one
layer's towers to the whole set. That is exactly what makes the memo survive the
step; it is not an incidental cost, and no correctness claim depends on it. The
risk it creates is capacity, not tokens: on a unified-memory box a model whose
host mapping and device allocations are both live could fail to fit, and such a
failure would present as an allocation error rather than as a slow number. That
risk was raised here before the fix was measured; the measurement below tested
it and it did not materialise on this artifact.
`AdoptDeviceBytesAsHost` is the lever that would collapse the host mirror onto
the device block, and IT DOES NOT APPLY TO THIS ARTIFACT. Its first branch needs
`w.mmap_src != nullptr && w.bytes.borrowed()`, and `OwnedTensor::mmap_src`
(`include/vllm/model_executor/models/qwen3_5_weights.h:119`) has exactly one
producer in the tree: `o.mmap_src = t.data;` at
`src/vllm/model_executor/models/qwen3_5_weights.cpp:517`, inside
`BorrowStTensorBytes` (`:491`), which takes an `StTensor` and is therefore the
SAFETENSORS direct-upload path. `grep -rn '\.mmap_src\s*=' src/ include/`
returns five hits and the other four are clears to `nullptr`. No GGUF path
assigns it, which is what the tree comment at `:265-266` already states and what
the header contract at `:112-118` turns on. The `MmapSrc(g, pol)` helper at
`qwen4_exp_weights.cpp:123` is NOT that field: it returns a `const GgufFile*`
whose only use inside `OwnGgufQuantBlocks` is `DropSpanResidency`, and the two
share nothing but a name. So on a GGUF weight adoption returns early, the host
mirror survives, and both copies are real.

**The focused gate is NOT `test_qwen4_exp_layer_loop` alone.** It must include
`test_qwen4_exp_moe` and `test_qwen4_exp_moe_sel_fp`. The reviewer's M5 mutation
deep-copied all six expert-tower entries instead of borrowing them, and the new
layer-loop file stayed fully green (470 of 470 assertions). The mutation was
caught only by the pre-existing W5d-4 cases, whose assertion is the borrow
identity itself -- `mw.expert_gate_kq.bytes.data() == s.w.gate_exps.bytes.data()`
(`tests/vllm/models/test_qwen4_exp_moe.cpp:567`). The borrow guarantee with the
largest blast radius, constraint (2) above and the whole of #2476, is pinned
there and nowhere else.

**The capacity risk was raised, measured, and did not materialise on this
artifact.** The paragraph above predicted that the released UD-IQ1_S artifact
would be held twice -- roughly 68 GiB of host mapping plus roughly 68 GiB of
device allocations -- and that the failure would present as an allocation error
rather than as a slow number. THAT PREDICTION IS FALSIFIED by the A/B below.
Peak resident memory is unchanged within 0.006% on both boxes, steady-state
resident memory is LOWER after the hoist, and no arm's server log contains
`out of memory`, `bad_alloc` or `cudaErrorMemoryAllocation`. The mechanism is
exactly as described -- the deleter moved from per-layer to model lifetime, and
that is what makes the memo survive the step -- but the predicted consequence
was wrong. What the measurement covers is ONE artifact (UD-IQ1_S, ~68 GiB) on
TWO unified-memory boxes of 122-128 GB. It does not license a general claim
that permanent device residency is free for every model and every checkpoint; a
larger artifact, or a box with a discrete device memory pool rather than a
unified one, is unmeasured and the axis stays open there.

### W6 measured: 33.4x, on the nsys-predicted kernel floor (2026-09-13)

Interleaved same-tree A/B. BASE is `3cafbcaf718816f8e60bde525d3c30bac1016a20`
(the profile commit, no hoist); FIX is
`72498897144afb4d7037e3c3f48a8c0f8b4f3223` (this change). Two rounds alternating
BASE/FIX, one server boot per arm, released `unsloth/Qwen3.8-Flash-Next-GGUF`
UD-IQ1_S staged to local disk, `--max-num-seqs 1 --device cuda` and no other
engine flag, 16-token decode, median inter-token delta.

| Box | Round | Arm | tok/s | s/token |
|---|---|---|---|---|
| `dgx:gpu0` (GB10, sm_121a) | 1 | BASE | 0.2570 | 3.88307 |
| `dgx:gpu0` | 1 | FIX | 8.5655 | 0.11689 |
| `dgx:gpu0` | 2 | BASE | 0.2570 | 3.88307 |
| `dgx:gpu0` | 2 | FIX | 8.5816 | 0.11685 |
| `thor:gpu0` (sm_110) | 1 | BASE | 0.2645 | - |
| `thor:gpu0` | 2 | BASE | 0.3004 | - |
| `thor:gpu0` | - | FIX | 4.9278 | - |

**33.4x on `dgx:gpu0`, reproduced in both rounds.** TTFT falls from 9.0 s to
1.23 s. The FIX arm's per-token spread across all 14 deltas is 0.1154-0.1180 s,
so the two rounds are not averaging over a bimodal step. `thor:gpu0` confirms
the direction on a second architecture; its BASE spreads 0.2645-0.3004 tok/s,
which is why the ratio is quoted from `dgx:gpu0` only.

**The measured step lands on the predicted floor, and THAT is what makes this a
confirmed diagnosis rather than a lucky fix.** The nsys profile measured
0.101 s of GPU kernel time per step against 3.69 s of allocator and copy time.
If the allocator was the whole bill, removing it must leave a step at roughly
0.10 s. The measured FIX step is 0.1169 s. The diagnosis, the fix and the trace
agree, so the remaining 0.016 s is the host-side cost the allocator was hiding
and is not an unexplained residue.

**Memory, which is the axis the section above owed.** `VmHWM` is peak resident
set size, read from `/proc/<pid>/status` of the server process.

| Box | Arm | `VmHWM` (kB) | steady `VmRSS` (kB) |
|---|---|---|---|
| `dgx:gpu0` | BASE | 77,397,988 | 77,397,988 |
| `dgx:gpu0` | FIX (round 1) | 77,402,288 | 43,977,844 |
| `dgx:gpu0` | FIX (round 2) | 77,398,924 | 43,603,596 |
| `thor:gpu0` | BASE | 77,304,396 | - |
| `thor:gpu0` | FIX | 77,305,524 | - |

Peak is identical within 0.006%, and steady-state resident set falls by roughly
33.4 GB on `dgx:gpu0`. No arm logged `out of memory`, `bad_alloc` or
`cudaErrorMemoryAllocation`. The predicted doubling is therefore not what this
box does: the host mapping and the device block are not two independent 68 GiB
charges against physical memory here, and the per-step staging BASE performed
was itself resident cost that the hoist removes.

### The decode-step profile, measured (2026-09-12, `dgx:gpu0`, `51c248190`)

Everything in the section below this one was a hypothesis. This section is the
measurement that ranks them, and it moves the answer off all of them.

**Harness.** `dgx:gpu0` under an `rc` lease, GB10 `sm_121a`, the released
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S staged to LOCAL disk, server at
`--max-num-seqs 1 --device cuda` and no other engine flag. Build: `Release`,
`VLLM_CPP_CUDA_ARCHITECTURES=121a`, tests OFF. COMPILED FEATURE SET ASSERTED:
`fa2` ENABLED for `[121a]` by the arch table but the FA2 manifest is `[]`,
because CUTLASS headers are absent and the fetch is opt-in
(`-DVLLM_CPP_CUTLASS_FETCH=ON`); `marlin-nvfp4` and `cutlass-nvfp4` resolve for
`[121a]`. NONE OF THE THREE IS REACHABLE BY THIS MODEL: `qwen4_exp` registers
only `kLinearAttention` (GDN) and `kQwenSparseAttention` (QSA), QSA has its own
kernel (`cuda_qwen4_exp_qsa.cu`), FA2 lives behind `cuda_paged_attn.cu` /
`cuda_ops.cu`, and Marlin/NVFP4 are fp4 paths while this artifact is GGUF
k-quant. The trace confirms it: no FA2 and no NVFP4 kernel appears.

**The speed, reproduced.** 0.25 tok/s, across four server boots and 150+ tokens:
3.99, 4.00, 4.05, 3.95 s per token, every individual delta inside 3.81-4.23 s.
The published 0.25 was not an artifact of a truncated stream; it is the steady
state.

**It is inside the forward.** `VT_ENGINE_STEP_LOG=1`: every token-producing step
is `model_executed=1` with `elapsed_s` 3.81-4.01, and every other step returns
in 0.000 s. 26 steps, 20 of which ran the model. Prefill of a 35-token prompt is
ONE step of 8.9 s, which is the TTFT. No timer, no poll, no stall.

**Where the forward goes.** `nsys`, 60 s window bracketing ~15.2 steady-state
decode steps, report written to LOCAL disk (the sqlite export fails on the CIFS
mount with "database is locked" and leaves a report with no `StringIds` table,
which `nsys stats` then reports as SKIPPED -- a failed export that reads like a
measurement of nothing):

| | total in window | per step | share |
|---|---|---|---|
| `cudaMalloc` | 34.195 s / 5,740 calls | 378 calls, **2.25 s** | 60.5% |
| `cudaMemcpyAsync` | 12.542 s / 19,878 calls | 1,308 calls, **0.82 s** | 22.2% |
| `cudaFree` | 9.416 s / 5,741 calls | 378 calls, **0.62 s** | 16.7% |
| `cudaLaunchKernel` | 0.297 s / 38,006 calls | 2,534 launches, 0.02 s | 0.5% |
| `cudaStreamSynchronize` | 0.016 s / 5,577 calls | 367 syncs, 0.001 s | ~0.0% |
| ALL GPU KERNELS | 1.540 s | **0.101 s** | **2.6%** |

Host-to-device traffic is 593,892 MB in the window -- 38 GiB per step over 846
copies, the largest 471.859 MB, which is exactly one `[512, 640, 2560]` IQ4_NL
expert tower.

**97% of a decode token is the CUDA allocator and the copies it feeds. 2.6% is
compute.**

**The 378 allocations per step account for themselves exactly.** 48 layers x
(3 expert towers + 3 shared-expert weights + router + shared gate) = 384. That
is `Qwen4ExpMoeBlockWeights`, rebuilt inside the layer loop
(`qwen4_exp_forward.cpp:673`), handing `ResidentWeight` fresh borrowed views
whose `d_dev` is null every step, so every tower is `cudaMalloc`ed, copied,
used and `cudaFree`d per step. A ~470 MB `cudaMalloc` costs 5.96 ms on average
here and up to 30 ms. `Backend::Alloc` on CUDA is a plain `cudaMalloc`; the
pooled `cudaMallocAsync` appears separately at 1,424 calls and is not this.

**The fix has a landed precedent twelve lines away.** GDN had the identical
defect and it was repaired under
[#2476](https://github.com/mudler/vllm.cpp/issues/2476): `lw.gdn_block` is an
`optional` built once and held by the model, and the comment there names the
failure -- a per-step adapter "re-established residency every step". The MoE
adapter is that bug, unfixed, on the same row.

**What this retires.** Launch overhead is 0.5% and the real count is 2,534, not
the withdrawn "~1,400". The MoE host round-trip costs 1 ms per step and is a
CAPTURABILITY defect, not a speed one. Copy bandwidth is 22%, not the bill.
"Weight aliasing over UMA" has the direction backwards: the weights DECLINE the
alias (the residency instrument reads `declined_borrow=96.958 GiB` against
`aliased_in_place=0.014 GiB`, because a GGUF mmap borrow owns no anonymous pages
and `MakeHostBytesDeviceAliasable` refuses it by name) and are therefore STAGED,
which is the expensive arm rather than the cheap one.

**Two experiments that measured nothing, recorded so they are not repeated.**
`VT_QWEN35_GROUPED_MOE=0` came out flat (4.05 vs 4.00 s) and was read as
refuting the tower-staging hypothesis. It cannot test it: the per-expert arm
reaches `KqResidentSlice`, whose first line is
`ResidentWeight(d, w)` over the WHOLE tower before the pointer arithmetic, so
both arms stage identical bytes. Separately, two `nsys` runs produced empty
reports -- one exported sqlite onto CIFS, one had its profiler SIGKILLed by the
harness's own shutdown, which writes no report at all.

**The floor this implies.** Kernel time is 101 ms per token, and the top kernel
is `HcGroupedNormKernel` at 40.7% (96 instances per step, ~435 us each, ~42 ms
per token). A memoised adapter puts the step in the neighbourhood of the kernel
floor; the exact landing point is NOT projected here, because removing the
allocator can expose host-side cost it currently hides. Re-measure, do not
assume.

### Candidate causes, and what is actually known about each

CORRECTED 2026-09-12
(`.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2AAC4XK43PJMEVTQVVDDS54.md`).
The three entries below landed as "known bottlenecks explaining the gap" and as
"causes identified". Two of them were readings of other people's comments, and
the third was a derivation. They are hypotheses, and this section now says so
beside each one. Nothing here has a profile behind it yet; the trace that would
rank them is owed.

1. **The GGUF k-quant MoE arm is host-mediated, and therefore uncapturable.**
   STRUCTURE READ FROM THE SOURCE, COST NOT MEASURED. `MoeBlock`
   (`qwen3_5.cpp:7178`) has three arms: NVFP4 fused, bf16 fused, and a
   "reference path" for everything else. A GGUF k-quant checkpoint has neither
   fp4 nor bf16 experts, so it takes the third, which copies the hidden state
   to host and synchronizes once per layer (`:7211-7213`), downloads the router
   top-k (`:7225-7226`), and makes three `KqGrouped` calls (`:6133-6145`) that
   each upload, launch one grouped GEMM and `Download` the f32 result, with the
   SwiGLU in a host loop between them. The tree states the consequence at
   `:7191-7193`: the fused arms are "capturable", the reference path is "not the
   capture target". So the missing decode graph is not a `qwen4_exp` property —
   it is a property of the whole GGUF MoE lane, and it is the corrected form of
   what entry 2 below used to claim. Filed as
   `.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2AAACGC0SXYXFN2JQ3GFEAZ.md`.
2. **The MoE weight adapter is rebuilt per layer per step, and the rebuild drops
   the device-residency memo.** STRUCTURE READ FROM THE SOURCE, COST NOT
   MEASURED, AND THE TWO POSSIBLE COSTS DIFFER BY ORDERS OF MAGNITUDE.
   `qwen4_exp_forward.cpp:673` composes `Qwen4ExpMoeBlockWeights` inside the
   layer loop, and `BorrowWholeOwnedTensor` (`qwen3_5_weights.cpp:340-368`)
   carries every field except `d_dev`. Whether the resulting re-derivation costs
   a cheap alias re-test or a full re-upload of the tower turns on a 256-byte
   alignment test (`qwen3_5_weights.cpp:252`) that source inspection cannot
   decide. Filed as
   `.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS.md`.
   The `#2336` citation this section used to carry is wrong and is withdrawn:
   #2336 is the PLE block, its gate op and the layer loop, and its one
   `ResidentWeight::d_dev` remark is about the GDN adapter.
3. **Weight residency over UMA.** DERIVED, NOT READ. "~136 GB working set" is
   an inference from the ~68 GB artifact and a doubling, not a measurement of
   this process; `DeviceStagingFits` is the predicate that decides whether the
   weights stage or stay aliased, and nobody read its answer on this run.

WITHDRAWN: "each decode step launches ~1,400 kernels, versus 1 for
`Qwen3.5ForCausalLM`". Nobody counted this tree's launches. The number is a
misreading of `qwen3_5.cpp:1096`, which says `ResidentWeight` "re-enters the
alias branch about 1,361 times per decode step" — weight-residency calls, on a
different checkpoint, in qwen3_5's host-alias instrument. A launch count needs a
profiler and no trace was taken.

None of these is a ceiling, and the first two are ordinary implementation
defects with named fixes.

### Mutation record — W10 (#1978, dead-end)

**The hypothesis.** `ISSUE-LOCAL-01M2TZ9AE416FW4GH23V09783Q` read the
slowest decode kernel (`QuantDotGemmGrouped32Kernel`, 41.6 GB/s, 15.2%
of peak) off the source and attributed the cost to the one-warp-per-output
shape: at `nb = 80` the inner loop is 2.5 iterations per lane, then a
five-step `__shfl_down_sync` where 31 of 32 lanes idle. The fix had a
landed precedent 120 lines below: `QuantDotGemmGroupedFusedSwiGLU32Kernel`
already amortises at width 2. Generalising to N columns per warp would make
the loop `2.5 × N` useful iterations per reduction and read the broadcast
activation once per warp.

**What landed.** `QuantDotGemmGrouped32MultiColKernel` — one warp computes
C consecutive output columns against ONE broadcast activation row, keeping C
accumulators and paying the 5-step reduction once for C results. Bit-identical
by construction: each output keeps its own accumulator and its own ascending
block order; no re-association.

`VT_V4_W32_COLS` (1/2/4/8, default 1) and `VT_V4_W32_WARPS` (1/2/4/8,
default 4) select the tile per call. `kW32ColsDefault` stays 1.

A byte-identity gate: 48 combinations (3 dtypes × 2 nb × 4 n × 2 bcast),
all compared by `==` (not tolerance), with a `kPoison` fill so a zero-grid
arm reads as FAILURE.

A sweep harness (`benchmarks/w32_grouped_cols_ab.cpp`) runs the released
decode shape (hidden=2560, inter=640, experts=512, rows=10, nb=80, IQ4_NL)
at all cols/warps arms, printing median μs + GB/s + FNV-1a bit hash.

**The RED, before the change.** The multi-column kernel did not exist; the
byte-identity gate had no target to compare against.

**The gate, and why each half discriminates.** The gate is `==` (byte-identical),
not a tolerance. The sweep harness prints an FNV-1a hash per arm; a non-identical
arm produces a different hash. The `kPoison` fill catches a zero-grid arm that
would otherwise compare equal to another zero-grid arm by accident.

**The battery.**

| suite | cases | assertions |
|---|---|---|
| `test_cuda_quant_dot` (32-block grouped multi-COLUMN GEMV) | 1/1 passed | **35,858** |

Build on `thor:gpu0` (sm_110, nvcc 13.0, `VLLM_CPP_CUDA_ARCHITECTURES=110`):
858/858 targets linked, `BUILD_EXIT=0`.

**Counts, before and after, on the same tree.** Before: 1 warp per output
element, `cols=1` only. After: 1/2/4/8 cols × 1/2/4/8 warps, all
byte-identical (same FNV-1a `0xce92c733fbd08728`), 35,858 assertions
across 48 combinations, 0 failed.

**The sweep, measured on `thor:gpu0` (sm_110), 500 iters / 50 warmup:**

```
shape hidden=2560 inter=640 experts=512 rows=10 nb=80 dtype=iq4_nl
tower=450.0 MiB  per-launch bytes=8.816 MiB (weights 8.789 + act 2720 B + out 25600 B)

cols   warps  blocks   median_us       GB/s   vs_cols1              bithash
1      4      1600         606.0       15.3      1.000   0xce92c733fbd08728
2      4      800          602.9       15.3      1.005   0xce92c733fbd08728
4      4      400          604.7       15.3      1.002   0xce92c733fbd08728
8      4      200          606.2       15.2      1.000   0xce92c733fbd08728
1      1      6400         605.4       15.3      1.001   0xce92c733fbd08728
1      2      3200         605.8       15.3      1.000   0xce92c733fbd08728
1      8      800          605.4       15.3      1.001   0xce92c733fbd08728
4      1      1600         605.6       15.3      1.001   0xce92c733fbd08728
4      2      800          605.9       15.3      1.000   0xce92c733fbd08728
4      8      200          604.9       15.3      1.002   0xce92c733fbd08728

OK: every arm produced byte-identical output.
```

All 10 arms within 0.5% of each other at ~605 μs / 15.3 GB/s. No speedup.

**Why it is a dead-end, measured rather than asserted.** The kernel is
memory-bound, not compute-bound. The numbers:

- 8.816 MiB per launch (8.789 MiB weights + 2720 B act + 25600 B out)
- 606 μs to move 8.816 MiB = 15.3 GB/s
- GB10 peak: ~273 GB/s (LPDDR5x unified memory)
- Utilization: 5.6%

The multi-column tile does not help because:

1. The activation row is 2720 bytes — already in L2, so broadcasting it to
   C columns saves nothing.
2. Weight reads dominate (99.7% of bytes) and are the same regardless of
   cols. Each of 512 expert groups has 10 rows (800 bytes) at a stride of
   `intermediate_size × nb_blocks` — a scattered-access pattern where the GPU
   waits on memory latency, not compute.
3. The warp reduction is 5 steps × ~1 cycle = 5 cycles, negligible against
   ~2000 cycles of memory latency per weight fetch.

The issue's own bound was honest about this: "NEITHER IS PREDICTED HERE.
The kernel may be bound by something this reading does not see." This sweep
is the measurement that sees it. The bottleneck is the memory system, not
the reduction tree.

**What would move this kernel.** Coalescing weight reads across warps (warps
in the same block read consecutive expert rows and share via shared memory),
or a persistent kernel that prefetches the next expert while computing the
current one. Both are larger changes than tile widening.

**What this retires.** `ISSUE-LOCAL-01M2TZ9AE416FW4GH23V09783Q`'s owed item
1 (the N sweep). The sweep ran; the answer is flat. `kW32ColsDefault` stays
1. The kernel and gate land as a correct-and-measured negative result.
