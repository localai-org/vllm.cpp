# Muse Glimmer — Meta's 30B open agentic multimodal model

**Row:** `MODEL-MUSE-GLIMMER`
**Issue:** [#268](https://github.com/mudler/vllm.cpp/issues/268)
**Base SHA:** `a0fa12c7219a86832412a6ece1490f452c1d1c40`
**Upstream anchor:** vLLM PR [#51655](https://github.com/vllm-project/vllm/pull/51655) head `075d645af`
**Secondary C++ reference:** llama.cpp PR [#26841](https://github.com/ggml-org/llama.cpp/pull/26841) (MERGED 2026-08-10)
**Checkpoint:** `meta-models/Muse-Glimmer-30B`, Apache-2.0, bf16

## 0. Honesty statement — what is and is not claimed

Three things are unusual about this row and none of them may be papered over.

**The anchor is not the pin.** Our parity pin `555967922` (2026-07-26) contains
no Muse Glimmer code — `grep -ril 'muse\|glimmer' vllm/model_executor/models/`
at the pin returns nothing — and neither does vLLM `main`. The only upstream
implementation is PR #51655, opened 2026-08-10T10:20Z, approved but **unmerged**,
with 3 of 20 CI checks red. This row therefore ports from a **branch head**, not
from the pin. That is a deliberate exception, taken on explicit developer
direction (2026-08-10), recorded as deviation 16 in
[`.agents/porting-inventory.md`](../porting-inventory.md) §9 and argued for in
the commit that introduces it, which is where exceptions live now that the
waiver registry is retired (`a4f72f86`). It owes a re-anchor when #51655
merges.

**There is no gateable oracle, so no speed number is claimable.** AGENTS.md
requires both sides of any throughput comparison to run the pinned oracle on
identical workloads, and requires an oracle to demonstrably build and run the
model before it is gateable. The pinned vLLM cannot load `muse_glimmer` at all,
and the checkpoint needs `transformers 5.15.0.dev0` against the pin's 5.14.1.
**Every performance axis for this model is an open gap** until the pin advances.
Correctness is gated against the HF reference implementation instead. No
"parity" or "faster than vLLM" claim may be made from this row's evidence.

**llama.cpp is a secondary reference only.** Per
[[vllm-is-the-bar-not-llamacpp]], llama.cpp informs C++ structure and is useful
for cross-checking dequant and tokenizer behaviour, but it is never the
correctness oracle and never the speed denominator. Anything taken from it is
cited as such and marked in the porting inventory.

## 1. Architecture

`MuseGlimmerForConditionalGeneration` → `MuseGlimmerForCausalLM`,
`model_type: muse_glimmer`. The registry maps both the conditional-generation
and causal-LM architecture strings onto one class
(`registry.py`, PR #51655).

### 1.1 Text tower

52 layers, hidden 6656, 32 query heads / 2 KV heads (GQA 16:1), head_dim 128,
vocab 202048, max position 131072, rope theta 500000, `hidden_activation` gated
MLP, no attention logit softcapping.

| Mechanism | Detail | Anchor |
|---|---|---|
| Sandwich norms | `input`, `post_attention`, `pre_feedforward`, `post_feedforward` RMSNorm, fp32 compute, **baked `+1` weight offset** | `muse_glimmer.py:1236-1247` |
| Split eps | pre-norms use `rms_norm_eps`; post-norms use a separate, smaller `post_norm_eps` | `muse_glimmer.py:1239-1246` |
| iRoPE | `no_rope_layers[i] == 1` → RoPE **and** sliding window; `== 0` → NoPE **and** full attention | `muse_glimmer.py:1114-1116`, `:1167-1168` |
| QK-norm | weightless RMSNorm over `head_dim`, fp32, applied **before** RoPE | `muse_glimmer.py:1189-1196` |
| Query pre-scale | `scale_query_by` (~3.87) multiplies q *after* QK-norm; softmax scaling stays `head_dim**-0.5` | `muse_glimmer.py:1112`, `:1192` |
| Attention output gate | `sigmoid(output_gate_proj(hidden_states)) * attn_out`, gate reads the **layer input** | `muse_glimmer.py:1203-1206` |

Two of these are correctness traps and get their own RED-first tests:

- **The query pre-scale has two config schemas.** The native `params.json` ships
  the raw `qk_scale_factor` (~43.784); the modular HF `text_config` ships it
  pre-folded by `1/sqrt(head_dim)` (~3.87). Upstream disambiguates *by
  magnitude* — `qk_scale >= sqrt(head_dim)` means native, divide; otherwise use
  as-is (`muse_glimmer.py:472-517`). Getting this wrong scales every query by
  11.3x and is not a subtle drift.
- **`use_qk_norm` and `use_attn_output_gate` read as `None`, not `True`,** in the
  modular schema. Muse Glimmer always applies both; only an explicit `False`
  disables them (`muse_glimmer.py:456-469`). A naive `getattr(..., False)`
  silently drops both mechanisms and still produces plausible text.

### 1.2 Perception encoder

50 layers, hidden 1536, 16 heads (head_dim 96), patch 14×14, `patch_temporal` 2,
`pos_emb` grid 32×32, interleaved `window_attention` / `full_attention` per
`layer_types`, projector 4096 → output 6144. Image token 200092, video token
200091; placeholder strings `<|patch|>` / `<|image|>` / `<|video|>`
(`muse_glimmer.py:127-129`).

| Stage | Detail | Anchor |
|---|---|---|
| Patchify + embed | `conv1_linear: Linear(patch_temporal*3*patch_size^2 → hidden, bias=False)` — a **linear on patchified input**, not a conv | `muse_glimmer.py:696`, `:710` |
| Positional embedding | learned `[32*32, hidden]`, **bilinear-interpolated** to the actual grid with per-corner validity masking | `muse_glimmer.py:761-820` |
| 2D RoPE | `spatial_dim = head_dim//2`, base 10000, `freqs = cat([freq_w, freq_h])` — **width first** | `muse_glimmer.py:741-759` |
| Window attention | blocks of `pos_emb_height × pos_emb_width`, `-1`-padded permutation, per-block valid counts become `seq_lens` | `muse_glimmer.py:844-867` |
| Pixel-shuffle downsample | by `merge_kernel_size`; asserts `output_dim == hidden * merge^2` | `muse_glimmer.py:822-842`, `:734-739` |
| ln_pre / ln_post | plain `LayerNorm` (not RMSNorm) | `muse_glimmer.py:714`, `:732` |

The width-before-height RoPE concatenation and the `+0.5 / -0.5` half-pixel
convention in the positional interpolation are both easy to transpose and both
produce a plausible-but-wrong image understanding. Each gets a fixture test.

### 1.3 DFlash drafter

The PR adds no new drafter. It reuses `qwen3_dflash`, recognising
`MuseGlimmerAssistantModel` as a `dflash` method (`config/speculative.py`), and
threads the **target's** `is_neox_style` into the draft config
(`qwen3_dflash.py:75-94`, `v1/spec_decode/dflash.py:83-95`). Upstream's own
comment is the important part: a RoPE-layout mismatch between draft and target
is **silent** — acceptance collapses, nothing errors, output stays correct. Our
DFlash row already exists ([[dflash-correctness-done-speed-bf16-blocked]]), so
this is a recognition-and-threading change, not a new speculator.

## 2. Reuse — the shared seams this must route through

A capability not reachable through the shared surface is not done. Every Muse
mechanism has an existing home:

| Muse piece | Existing seam | Files |
|---|---|---|
| Sandwich norms, `+1` offset, fp32 | Gemma 2/3/4 norm path | `gemma2.cpp`, `gemma4.cpp` |
| Attention output gate | Qwen3.5 gated attention | `qwen3_5.cpp` |
| Weightless QK-norm | Qwen3 family | `qwen3_vl_text.cpp` |
| Windowed-attention vision tower | Qwen3-VL vision seam | `qwen3_vl_vision.cpp` |
| Gated MLP | `layers::MlpGateUpMethodBase`, `vt::MergedGemmGroup` | — |
| Decode | `ModelRegistry::Forward`, `dense_attn::AttnBlock`, on-device sampling | — |
| Fusion | `vt::FusedChain` | — |
| DFlash | existing speculator path | — |

**iRoPE / per-layer NoPE has no analogue** — we have no Llama-4 — and is the one
genuinely new text-side mechanism. It extends the existing per-layer attention
config rather than introducing a parallel path; if the current layer-type seam
cannot express "NoPE + full attention" alongside "RoPE + sliding", the seam is
extended, not bypassed.

Everything ships as **additive files** mirroring vLLM's structure:
`muse_glimmer.cpp`, `muse_glimmer_text.cpp`, `muse_glimmer_vision.cpp`,
`muse_glimmer_weights.cpp`, `muse_glimmer_registry.cpp`, plus
`include/vllm/model_executor/models/muse_glimmer.h`. The capability is exposed
through `include/vllm.h`; examples and the server stay thin ABI clients
([[examples-are-abi-clients-only]]).

## 3. Work breakdown

| W | Scope | Gate |
|---|---|---|
| **W0** ✅ | Config parse + registry + weight map. No forward. | **LANDED 2026-08-10**, see §8 |
| **W1** | Text tower forward: sandwich norms, iRoPE/NoPE, QK-norm, query pre-scale, output gate | Per-layer reference-dump match, RED-first per mechanism |
| **W2** | Text e2e greedy vs HF reference | Token-exact on a fixed prompt set |
| **W3** | Perception encoder: patchify, pos-emb interp, 2D RoPE, window attention, pixel shuffle | Tower dump match, per-stage fixtures |
| **W4** | Image e2e (placeholder expansion, projector, scatter into text) | STRICT vs HF reference |
| **W5** | Video (`patch_temporal`, frame sampling) | STRICT vs HF reference |
| **W6** | DFlash drafter recognition + `is_neox_style` threading | Acceptance-rate check vs spec-off; drafts must actually be consumed |
| **W7** | Reasoning + tool parsers on the server surface | Ported upstream parser tests |

W0–W2 are the critical path. W3–W5 depend on the vision seam. W6 must force
async off on both arms ([[engine-proc-dropped-every-speculator-draft]]) or the
A/B is meaningless.

## 4. Gates

**Correctness, against the HF reference** (not the pinned oracle, which cannot
run this model — see §0):

1. Config parse: both schemas resolve to the same `scale_query_by` (~3.87), and
   `use_qk_norm` / `use_attn_output_gate` default **on**.
2. Per-mechanism RED-first: disabling QK-norm, the output gate, the query
   pre-scale, the `+1` norm offset, or the NoPE/RoPE layer split must each turn
   its test red. Mutation is the proof, not inspection.
3. Text e2e: token-exact greedy vs the HF reference on a fixed prompt set.
4. Vision: per-stage tower dumps within the bf16-depth envelope; image and video
   e2e STRICT.
5. DFlash: drafts demonstrably consumed (not silently dropped), acceptance rate
   recorded, output identical to spec-off.

**Ported tests.** The upstream tests in PR #51655 —
`tests/transformers_utils/test_muse_glimmer_config.py`,
`test_muse_glimmer_config_schema_norm.py`, and the five
`tests/tool_use/test_muse_glimmer_*.py` — are ported in the same change with
parameters, fixtures, tolerances and failure cases preserved, each carrying the
`075d645af` revision anchor. Harness adaptation is documented where unavoidable.

**Speed: OPEN GAP, not measured, not waived.** Recorded in `docs/BENCHMARKS.md`
as pending on the named external unblocker: #51655 merging and the pin advancing
to include it plus transformers 5.15. No ceiling is declared
([[when-stuck-workflow-scan-vllm]]). That row LANDED 2026-08-11 in the
at-a-glance table; it states no number, claims no waiver, and names the
unblocker.

`docs/STATUS.md` and `.agents/NOW.md` carry NO Muse row, and that is a stated
omission rather than an oversight. `STATUS.md` sits byte-exactly on its
shrink-only ratchet (243,451 of 243,451) and `NOW.md` at 5,986 of its 6,000
characters, so either page can only take a Muse row by collapsing another row's
binding narrative — somebody else's keyed record, which this round will not
rewrite as a side effect. Both are owed by the landing commit, which can pay for
them; until then the model's public state lives in `docs/FEATURES.md` and
`docs/USAGE.md`.

## 5. Risks

- **The anchor can change under us.** #51655 is an open branch with red CI; a
  force-push or review round rewrites what we cite. Mitigation: cite the exact
  head `075d645af`, keep the fetched ref, and diff before every re-anchor.
- **Unmerged upstream may itself be wrong.** The approved-but-red state means
  upstream's own gates have not fully passed. Where our HF-reference gate
  disagrees with PR #51655, the HF reference wins and the divergence is reported
  upstream rather than silently mirrored.
- **No oracle means correctness rests on one leg.** The HF reference is the only
  cross-check until the pin moves. Per-mechanism mutation testing carries more
  weight than usual here.
- **Weights are not local.** ~60 GB bf16. Download needs explicit authority.
- **GB10 memory.** 30B bf16 will not fit comfortably alongside anything else;
  `gpu_memory_utilization` reserves host RAM on unified memory
  ([[gb10-unified-memory-oom-reboots-box]]). Quantized arms first where possible.

## 6. Stop conditions

- #51655 is force-pushed or substantially rewritten → stop, re-diff, re-anchor
  before continuing.
- The HF reference and PR #51655 disagree on a mechanism → `NEEDS_DECISION`,
  do not guess which is authoritative.
- Any request to state a speed number for this model before the pin advances →
  refuse; the axis is an open gap by construction.
- Weight download or GPU time beyond what is already authorised → stop and ask.

## 6.5 REAL-WEIGHTS RESULT — our text tower agrees with an independent
transcription on the released checkpoint (2026-08-10)

The first evidence for this model that comes from the actual released weights
rather than a synthetic fixture.

**Reference.** `scripts/mm/muse_glimmer_text_ref.py` transcribes the upstream
forward (vllm#51655 head `075d645af`) into plain torch, with no `transformers`
and no `vllm` import, and runs it on the checkpoint's own safetensors. It exists
because nothing on either box can load this model: released transformers does
not register `model_type: muse_glimmer`, the checkpoint ships no remote-code
file, and the parity pin has no `muse_glimmer` at all.

**Full-depth reference sanity (52 layers, real weights).** Prompt "The capital of
France is" produces argmax `[" key", " of", " the", " is", " Paris"]`, last
position top-2 17.086 vs 13.445, `|logit|` max 18.66 under the 20.0 soft cap. A
coherent, correct continuation is good evidence the transcription is faithful.
It also independently confirms two things W0/W1 had to infer: `scale_query_by`
resolves to 3.87, and the derived iRoPE mask agrees with the checkpoint's own
`layer_types` / `layer_rope_theta` encoding.

**Our C++ vs that reference (reduced depth 4/52, real weights).**

| Measure | Result |
|---|---|
| Tensor accounting | **51 / 51** enumerated names present |
| argmax | `27389 110709 32485 122967 152652` — **identical both sides** |
| max abs logit difference | 0.0889745 |
| cosine | 0.999981 |
| assertions | **1,010,282 passed**, 0 failed |

**A verification trap worth recording.** With the env gates unset the same binary
reports "3 test cases, 3 passed, 15 assertions" — it looks green while comparing
nothing, because the heavy cases return early and doctest counts them as passed
rather than skipped. The assertion count is the tell: a real comparison is over a
million assertions, a no-op is fifteen. Never read this gate's pass/fail without
reading its assertion count.

**What this is NOT.** Not token-exact against the model's own runtime — no such
runtime exists here to be exact against, so this is agreement with an independent
transcription of the same upstream source, which is a weaker claim than a true
oracle gate and is recorded as such. Not a full-depth comparison of our forward
(the 52-layer arm needs ~55.7 GB resident and is a named residual). Not any kind
of speed result; no denominator exists.

## 6.6 A REAL HF REFERENCE EXISTS — and we match it (2026-08-10)

§0 said no gateable reference existed. **That is now out of date**, and this
section supersedes it for correctness (the speed statement in §0 still stands
unchanged).

HF's Muse Glimmer implementation lives on the `exportable-muse` branch of
`huggingface/transformers` (`a9e337e8`, version 5.16.0.dev0) — *not* the
`new-model-addition-onyx` name vllm#51655's comment gives, which exists neither
as a branch nor as a repo. The branch carries the full
`models/muse_glimmer/{configuration,modeling,processing,image_processing,video_processing}`
set plus `muse_glimmer_assistant` (the DFlash drafter).

Installed on dgx at `$HOME/venvs/muse-onyx`, deliberately isolated: the pinned
oracle venv is untouched, and the new venv reuses the oracle's torch through a
`.pth` fallback rather than pulling a second CUDA build. `muse_glimmer` registers
in `CONFIG_MAPPING_NAMES` and `AutoConfig` resolves the released 52-layer config.
Note it registers under the multimodal AutoModel classes, so
`AutoModelForCausalLM` rejects it — use `MuseGlimmerForConditionalGeneration`.

**Result at reduced depth (4/52), real weights, prompt "The capital of France is":**

| Comparison | argmax | max abs diff | cosine |
|---|---|---|---|
| HF reference vs our torch transcription | **identical** | 0.117212 | 0.99997157 |
| our C++ vs our torch transcription | **identical** | 0.0889745 | 0.999981 |

Three independent implementations — HF's own, our transcription, and our C++ —
agree on `[27389, 110709, 32485, 122967, 152652]`. That is materially stronger
than §6.5's claim: the earlier evidence was agreement between two things we
wrote from the same source, and this adds a genuine third-party oracle.

**Still owed.** The full 52-layer HF comparison did not run: dgx had a live
`vllm serve` holding most of the 119 GB unified pool, leaving ~37 GB against a
60 GB load, and on GB10 an OOM can reboot the box. That job also did NOT hold
`${GPU_LOCK}`, so the mutex protected nothing — the reduced-depth run was chosen
to stay bounded. Re-run at full depth on an idle box.

Also still true: the perception encoder has no reference check, nothing has run
end to end through the server, multi-step decode and the sliding window are
untested, and **no speed axis is claimable on any dimension.**

## 6.7 OPEN GAP — the ATEM parsers do not see their own framing at server defaults

Found in the PR #279 review (2026-08-11) and **not fixed**; recorded here because
it is the difference between "the parsers are ported" and "channel scoping
works".

Upstream's `MuseGlimmerToolParser.adjust_request` (`:206`) and
`MuseGlimmerReasoningParser.adjust_request` (`:117`) force
`skip_special_tokens=False` on any request routed to Muse Glimmer. Both were
dropped in the port, justified in the shipped headers by the claim that the C++
seam "carries no `skip_special_tokens`". **That claim was false.**

- `skip_special_tokens` is declared `= true` on both request types
  (`include/vllm/entrypoints/openai/protocol.h:240` completion, `:461` chat) and
  honoured at `src/vllm/v1/engine/detokenizer.cpp:68`.
- The released checkpoint marks the ATEM framing markers special:
  `tokenizer_config.json` lists them under `extra_special_tokens`, and
  `tokenizer.json` carries `"special": true` for `<|eom|>` 200007, `<|eot|>`
  200008, `<|start|>` 200022, `<|message|>` 200023.

So at server defaults the detokenizer strips `<|start|>assistant to=…<|message|>`
before either parser runs. Channel scoping — the mechanism this port was built
around, the thing that keeps an `<atem:invoke>` echoed inside a `to=self`
reasoning block from becoming a tool call — cannot function, and raw ATEM markup
can fall through to the client. The parsers' own unit gates pass because they are
handed framed strings directly, which is exactly why this was invisible.

**Why it is not fixed here.** The gap is in the SEAM, not in the port: there is no
`adjust_request` dispatch site anywhere in the tree. `KimiK2ToolParser::adjust_request`
(`src/vllm/entrypoints/openai/tool_parsers/kimi_k2.cpp:87`) exists and has no
callers either, and several other parser headers record the same drop. Building a
request-mutation hook onto the shared `ToolParser`/`ReasoningParser` surface is a
seam change owned by no Muse row, and doing it inside a review-findings fix would
be exactly the "hand-roll a parallel path" AGENTS.md forbids. It is recorded as
visible debt in the two parser headers, `docs/FEATURES.md` and `docs/USAGE.md`,
and it owes its own issue and row.

`supports_required_and_named = False` (`tool parser :192`) is dropped for a
related reason: the seam has no named/required-`tool_choice` fast path to opt out
of today. If one is ever added it must read this flag, or Muse Glimmer will be
handed a path upstream explicitly refuses.

## 6.8 PR #279 review findings — what the fix round closed (2026-08-11)

`CLAIM-MUSE-GLIMMER-FIX`, `row/MODEL-MUSE-GLIMMER-FIX`, off reviewed head
`774c44d8`. The review's verdict was PASS-WITH-FINDINGS. Three findings were
COVERAGE HOLES — mutations that stayed GREEN, meaning a shipped guarantee had no
test at all — and those are the ones worth recording.

| Mutation | Before | After |
|---|---|---|
| `perception_emb_norm` condition INVERTED (`muse_glimmer_mm.cpp:217`) | wiring gate GREEN | wiring 8/9 pass, **18 assertions RED** |
| `perception_emb_norm` call DROPPED | wiring gate GREEN | wiring 8/9 pass, **17 assertions RED** |
| RoPE base hardcoded 10000 instead of `g.rope_theta` (5e5) | text gate GREEN | text 20/21 pass, **4 assertions RED** |
| RoPE base hardcoded 2.0 (the absurd control) | text gate RED | text gate RED (2 cases) |
| `else vt::RmsNorm(...)` input-layernorm fallback given the wrong eps | default arm **21/21 GREEN** | fallback arm **1 assertion RED** |
| `else vt::RmsNorm(...)` final-norm fallback given `gemma=true` | default arm **21/21 GREEN** | fallback arm **59 assertions RED** |

In every case the tree was restored from an in-memory snapshot and `git diff` on
`src/` verified empty before re-running green.

What closed them:

1. **`perception_emb_norm` had no test at all.** No config in the tree set
   `normalize_tok_embeddings`, and the scatter case compared merged rows against
   the same `soft` vector it had just computed, so it was structurally blind to
   what `EncodePixelGroups` did to that vector. The new case runs the IDENTICAL
   tower twice — flag off and flag on — which makes the norm the only difference,
   and requires the soft tokens to stand in the exact algebraic relation the
   weightless RMSNorm defines, with a control asserting the norm is not a no-op at
   this geometry.
2. **The whole non-FusedChain fallback arm was dead.** `FusedChainAdoptEnabled()`
   defaults ON and is read once per process into a function-local static, so the
   three `else vt::RmsNorm(...)` branches never executed under test and a
   same-process env flip could not reach them either. `tests/CMakeLists.txt` now
   registers the text binary a SECOND time as `test_muse_glimmer_text_fallback`
   with `VT_FUSED_CHAIN_ADOPT=0`, so the whole gate — including the fp32 reference
   comparison — runs on both arms, and a case inside the binary asserts the arm it
   actually took, so the second registration cannot silently repeat the first.
3. **RoPE theta was ungated at a realistic magnitude.** Every case ran at
   positions 0..4 with `head_dim` 4, where 5e5 and 1e4 differ by milliradians —
   under the reference band. The new case moves to positions 0..4096, pins the
   base three ways (matches its own reference, does NOT match a 1e4 reference,
   and the two configs cannot produce the same logits), and carries a widened but
   named band: measured 6.9e-4 against a wrong-base signal of 4.1e-2.
4. **A stale "OPEN FINDING" had disarmed a live assertion.** The real-weights
   gate demoted `CHECK(accounted == enumerated)` to a `MESSAGE` whenever
   `vision.present`, over a 50-tensor shortfall the W4 enumeration correction had
   already fixed. Re-armed; the same guarantee runs without the NAS in
   `test_muse_glimmer_wiring`, which asserts it on a multimodal checkpoint.
5. **Two shipped headers carried a false justification** for dropping
   `adjust_request`. Corrected, and the real consequence recorded as §6.7 above.
6. **`docs/USAGE.md` overclaimed and contradicted `docs/FEATURES.md`.** "matches
   it token for token on the released checkpoint" is now what was actually
   measured: reduced depth 4 of 52, five prefill argmax positions, not generated
   tokens, against transcriptions rather than the model's own runtime.

Deliberately left open: the `adjust_request` seam (§6.7, needs its own issue and
row); the `STATUS.md` / `NOW.md` rows (§4, blocked on their shrink-only budgets);
the lifecycle-token advance in `.agents/model-matrix.md`, which owes those two
rows; and the compact-vs-spaced `arguments` JSON, which is the whole parser
family's convention and is documented rather than changed in one parser.

## 7. Outcome

Pending overall.

## 8. W0 — the CPU scaffold (2026-08-10, `CLAIM-MUSE-GLIMMER-W0`)

Additive only: `include/vllm/model_executor/models/muse_glimmer.h`,
`src/vllm/model_executor/models/muse_glimmer{,_weights,_registry}.cpp`,
`tests/vllm/models/test_muse_glimmer_scaffold.cpp`. No forward, no checkpoint,
no GPU, no download.

Both architecture strings register onto one factory, mirroring upstream. The
config parse handles the canonical nested layout *and* normalizes the older flat
layout. The weight-name mapper ports `hf_to_vllm_mapper` for both checkpoint
conventions. The structural enumeration deliberately omits the three weightless
modules (`embed_norm`, the per-head `qk_norm`, `perception_emb_norm`) that ship
no tensor — enumerating them would make the loader demand tensors that do not
exist in any checkpoint. The forward refuses by name.

**Gate:** `test_muse_glimmer_scaffold` 11/11 cases, 73/73 assertions, clean CPU
`-Werror` build. Full CPU `ctest` green (regression: the change is additive TUs
plus two registry entries).

**RED-first mutation evidence.** Each of the four named traps was mutated in
tree, rebuilt, and confirmed to turn the gate red; the tree was then restored
byte-for-byte (verified by an empty `git diff`) and the gate re-run green:

| Mutation | Result |
|---|---|
| Native raw `qk_scale_factor` treated as already-folded (the 11.3x query blow-up) | 3 assertions RED |
| Absent `use_qk_norm` / `use_attn_output_gate` defaulted to `false` | 4 assertions RED |
| iRoPE mask counted forward instead of backward from the last layer | 5 assertions RED |
| Legacy sandwich-norm renames applied in the wrong order (swapping post-attention with pre-feedforward) | 1 assertion RED |

## 9. W1 readiness — the primitive map (surveyed 2026-08-10, no code yet)

Every primitive the text tower needs already exists, so W1 is a mechanical port
against `gemma2.cpp` (407 lines) rather than new kernel work. `gemma2.cpp` is the
template: its sandwich-norm decoder layer is structurally identical to Muse's.

| Muse mechanism | Existing primitive | Note |
|---|---|---|
| Sandwich norms with baked `+1` | `vt::RmsNormArgs{eps, gemma=true}` | `gemma=true` IS the `(1+w)` offset |
| Split pre/post eps | two `RmsNormArgs` values | the ONE delta vs gemma2, which uses a single eps |
| SwiGLU MLP | `layers::UnquantizedMlpGateUpMethod` | gemma2 uses the `...GeluMethod` sibling; Muse is silu |
| Attention output gate | `vt::kSigmoidGateBf16` (FusedChain) | already used by Qwen3.5; `attn * sigmoid(gate)` |
| Sliding vs full attention | `vt::PagedAttentionArgs::window_size` | gemma2 threads this per layer already |
| RoPE | `vt::RopeNeox` | skip entirely on NoPE layers |
| Weightless QK-norm / embed_norm | `vt::RmsNorm` with a ones weight | no weightless variant exists; a ones buffer is the cheap path |

Deltas from the `gemma2.cpp` template, each a place to get it wrong:

1. **Embedding scale.** Gemma multiplies by `sqrt(hidden)`; Muse instead applies a
   weightless RMSNorm (`embed_norm`, `muse_glimmer.py:1286`). Different operation,
   same slot.
2. **Split eps.** Pre-norms take `rms_norm_eps`, post-norms `post_norm_eps`.
   gemma2 threads one eps everywhere.
3. **Attention scale.** Muse uses plain `head_dim**-0.5`, NOT gemma2's
   `query_pre_attn_scalar**-0.5`; the query pre-scale is applied separately to q
   after QK-norm.
4. **iRoPE.** RoPE and sliding-window travel TOGETHER on `no_rope_layers[l]==1`;
   NoPE layers are full attention. gemma2's sliding split is independent of RoPE.
5. **The gate reads the normed layer input** (`dhn`), not the attention output.
6. **`lm_head` is untied** and there is an `output_multiplier` before the final
   soft-cap.

The post-attention and post-feedforward norms must stay STANDALONE, exactly as
gemma2 documents: they are sublayer-output norms with no residual add, so folding
them onto `kFusedAddRmsNorm` would be an incorrect fold.

**What W0 does NOT establish.** No forward runs, so nothing here says the model
produces correct tokens. The KV-cache spec is a documented placeholder: the real
sliding/full split rides the Gemma-4 per-layer spec seam and lands with W1. And
per §0 no speed axis is measurable at all while the pin lacks `muse_glimmer`. Filled in when the row reaches `DONE`: what was measured, what was
rejected and why, and why each default is set the way it is.

## 10. The GGUF k-quant arm (2026-08-11, `row/MODEL-MUSE-GLIMMER-GGUF`)

**Issue:** [#329](https://github.com/mudler/vllm.cpp/issues/329). Implements the
quantized-arm rule added to `AGENTS.md` in [#318](https://github.com/mudler/vllm.cpp/issues/318).

`LoadMuseGlimmer` used to throw `"does not support GGUF weights"`. That was never
a decision — the quantized arm simply was not on any list — and
the `.agents/porting-a-model.md` checklist §2 now makes it a rule: a
model port covers the quantized arms. A 30B bf16 checkpoint is ~60 GB against a
~17 GB k-quant, so for this model the k-quant is the arm most users can run at
all, and it is what a quant-matched llama.cpp comparison would need.

**Assets.** `meta-models/Muse-Glimmer-30B-GGUF` @ `2fb01e4e6f`:
`muse-glimmer-30B-kquant-17gb.gguf` (16.76 GB, arch `muse-glimmer`, 731 tensors),
`muse-glimmer-30B-kquant-dynamic.gguf` (19.65 GB, same 731, mixed per-tensor
types), `mmproj-kquant.gguf` (1.40 GB, arch `clip`, 809), `dflash-kquant.gguf`
(1.63 GB, arch `dflash`, 58).

### 10.1 Four convert-time transforms, each verified against the bf16 checkpoint

Derived from the GGUF's own tensor list and metadata, then cross-checked
element-by-element against `meta-models/Muse-Glimmer-30B` — not inferred from
names.

1. **The sandwich norms are stored PRE-OFFSET.** GGUF
   `blk.0.attn_norm.weight[0..5]` = `1.09619141, 1.11279297, 1.35742188, ...`;
   safetensors `layers.0.input_layernorm.weight[0..5]` =
   `0.09619141, 0.11279297, 0.35742188, ...`. Exactly `w_hf + 1`, and the same
   for all four sandwich norms. Our forward adds the `+1` itself
   (`RmsNormArgs{gemma=true}`), so the loader **subtracts one**. `output_norm`
   (the final norm) takes no offset in the model, is stored raw, and must NOT be
   un-shifted — the released checkpoint's `norm.weight` runs ±5 with mean 0.017,
   which is visibly not an offset weight.
2. **The query pre-scale is folded into `attn_q_norm`.** ggml has no weightless
   RMSNorm, so the converter materializes both weightless norms as vectors:
   `blk.N.attn_k_norm.weight` is all `1.0`, and `blk.N.attn_q_norm.weight` is the
   constant `3.87` — exactly `text_config.qk_scale_factor` in the safetensors
   `config.json`. The GGUF carries **no metadata key** for it, so
   `scale_query_by` is recovered from that tensor, checked constant on every one
   of the 52 layers, with a non-constant q-norm or a non-ones k-norm refused
   rather than averaged.
3. **The iRoPE mask rides `attention.sliding_window_pattern`** (52 bools,
   `true, true, true, false, ...`): `true` = RoPE + sliding, `false` = NoPE +
   full. Agrees with the safetensors config's `layer_types` and
   `layer_rope_theta` (NoPE at 3, 7, ... 51).
4. **`attn_q` / `attn_k` are stored in ggml's INTERLEAVED-RoPE row order**
   (`LlamaModel.permute`, llama.cpp `conversion/llama.py:163-169`, with `n_head`
   on the query side and `n_head_kv` on the key side), so the loader un-permutes
   both on the way into the merged `qkv_proj`. Added 2026-08-11: this one was
   MISSING and is the whole of issue #359 — see §13.

### 10.2 What is kept quantized, and what is not

| Operand | Residency | Why |
|---|---|---|
| `o_proj`, `output_gate_proj`, `down_proj` | **keep-quant** | standalone `[N,K]` MatmulBT operands, taken verbatim |
| `gate_up_proj` | **keep-quant block concat** when `ffn_gate`/`ffn_up` share a ggml type (they do: both Q4_K) | a k-quant row is a whole number of superblocks, so the merge is a byte concatenation |
| `qkv_proj` | **dequantized** | the forward wants ONE merged operand and the file's shards differ in type (`attn_v` Q6_K vs `attn_q`/`attn_k` Q4_K); two block encodings cannot share one tensor |
| `lm_head` | **dequantized** | consumed via `vt::Matmul` in Matmul-B `[H, vocab]`; a block encoding cannot be transposed without requantizing |
| `embed_tokens` | **dequantized** | a `[vocab, H]` gather table, not a GEMM operand |
| all norms | **dequantized** | `[H]`/`[Dh]` F32 vectors carrying a `-1` value transform |

### 10.3 Named residual — `post_norm_eps` (CLOSED 2026-08-11, see §15)

**Superseded.** This section used to read:

> The GGUF carries one epsilon (`attention.layer_norm_rms_epsilon` = 1e-5) and no
> post-norm key; the safetensors config ships `post_norm_eps` = 1e-8 separately, so
> the GGUF arm falls back to `rms_norm_eps` for the post-norms. Inside
> `1/sqrt(mean_square + eps)` with a mean square of order 1 that is a ~5e-6
> relative change — two orders of magnitude below bf16's ~4e-3 spacing, so it is
> not representable in the activation dtype. A real difference from the safetensors
> arm, just not an observable one. Fixing it needs a converter key upstream.

Two things were wrong with that. It did **not** need a converter key: 1e-8 is the
architecture's own constant in both references, so the right fallback was always
available locally. And the "not observable" conclusion rested on an assumed
post-norm input mean square of order 1 that nothing ever measured — these norms
take a *sublayer output*, and the sensitivity goes as `1/ms`. §15 closes it.

### 10.4 REFUSED and OWED — the mmproj perception encoder

`mmproj-kquant.gguf` maps cleanly for every tower tensor
(`v.blk.N.{ln1,ln2,attn_q,attn_k,attn_v,attn_out,ffn_up,ffn_down}`,
`v.{pre_ln,post_ln,position_embd}`, `mm.{0,1,2}` → adapter fc1/fc2 +
`vision_projection`) **except one**: `v.patch_embd.weight` is ggml ne
`[14, 14, 3, 1536]` = torch `[1536, 588]`, while `conv1_linear` needs
`patch_temporal * 3 * patch_size^2` = `2*3*14*14` = **1176** input features — and
the safetensors ships exactly `[1536, 1176]`. The `patch_temporal` axis is
absent, i.e. half the patch embedding does not exist in the file to be loaded.

The arm therefore **refuses by name** (`MuseGlimmerRefuseMmproj`) rather than
inventing a temporal half, and image/video keep using the bf16 safetensors. This
is **OWED**, not closed: the fix is upstream in the llama.cpp converter, and the
refusal should be retired the moment a converted mmproj carries the full weight.

### 10.5 The DFlash drafter — reachable, not exercised

`dflash-kquant.gguf` is arch `dflash` and every one of its 58 tensors is covered
by the ALREADY-LANDED `qwen3_dflash_gguf` name map (`fc.weight`,
`enc.output_norm.weight`, `output_norm.weight`, and 11 per block × 5 blocks). It
ships neither `token_embd` nor `output`, which is correct: a DFlash draft runs
the TARGET's embedding table and head, and the text GGUF ships both, so
`LoadGgufSharedEmbedAndHeadBf16` is the right source. **No new seam is needed.**
That is a structural reachability claim only — nothing here says a Muse Glimmer
draft proposes useful tokens; an acceptance-rate A/B is OWED and needs hardware
this row did not have.

### 10.6 What this arm does and does not establish

**Established.** The released 16.76 GB k-quant loads: 731/731 tensors accounted
with zero unaccounted in both directions, all 52 layers materialized at the right
shapes and orientations, the sandwich norms un-shifted (layer-0
`input_layernorm` min = −1.0, matching the safetensors' own min), and the query
pre-scale recovered as 3.87 across every layer. The 19.65 GB mixed-type
`dynamic` file accounts identically. Gate
`tests/vllm/models/test_muse_glimmer_gguf.cpp`: 12/12 cases, 428 assertions
with `VLLM_MUSE_GGUF`, 636 on the full-load case, with **11 mutations each
proven RED** and the tree restored byte-for-byte.

**SUPERSEDED (2026-08-11, §13).** "No forward was run on GGUF weights" held
when this section was written; a forward now runs and generates coherent text.
The structural result above was true AND insufficient — it could not see the Q/K
row permutation of transform 4, which preserves every name, shape, count and
dtype it checks.

**NOT established.** There is still no token-exactness against anything, and no
claim that the k-quant and the bf16 arm agree numerically — that comparison is
OWED (§13.4). And per §0 there is still **no speed
axis of any kind**: the pinned oracle cannot load `muse_glimmer` in either weight
format, so there is no denominator, and none is claimed.

## 11. The speed attempt (2026-08-11, `row/MUSE-BENCH`, issue [#333](https://github.com/mudler/vllm.cpp/issues/333))

**Result: no binding speed number was produced, and one bar moved from "assumed
reachable" to "blocked with a named cause."** Each cell below is either a value
with its arm stated, or a blocker. Nothing here is a ratio, because no cell has
two quant-matched sides.

| Bar | Arm | Outcome |
|---|---|---|
| **vLLM** | any | **OPEN GAP by construction.** Unchanged from §0: the pin carries no `muse_glimmer`, so there is nothing to divide by. Not waived, not substituted. |
| **llama.cpp** (SECONDARY) | `muse-glimmer-30B-kquant-17gb.gguf` | **runs**; one contended datapoint, below. **Non-binding** — the box was at load 39-123. |
| **ours** | same GGUF | **BLOCKED before the forward** — the tokenizer, not the loader. See §11.1. |
| **HF transformers** | bf16 safetensors | **BLOCKED**: needs dgx (`$HOME/venvs/muse-onyx`), whose GPU lock was held for the whole window by another session's 27B online-serving gate. Not interfered with. |
| **ours** | bf16 safetensors | **BLOCKED**: same host, and additionally needs a CUDA build that was deliberately not started so it would not perturb that measurement. |

### 11.1 Our GGUF arm cannot generate — the blocker is the TOKENIZER

§10.6 recorded "no forward was run on GGUF weights." The reason it cannot yet be
run is upstream of the forward and was not previously named:

```
vllm-bench --model muse-glimmer-30B-kquant-17gb.gguf ...
  -> failed: tokenizer: unsupported tokenizer.ggml.pre "llama4"
```

The file declares `tokenizer.ggml.model = gpt2` (which we accept) and
`tokenizer.ggml.pre = llama4`, which `Tokenizer::FromGguf`
(`src/vllm/tokenizer/tokenizer.cpp:766`) refuses by name.

**This is not an alias away.** llama.cpp maps `llama4` to
`LLAMA_VOCAB_PRE_TYPE_GPT4O` with `clean_spaces = false`
(`src/llama-vocab.cpp:2294-2299` @ `030ebb5`), and that pre-type carries its own
regex pair (`llama-vocab.cpp:428-434`) which is neither our `kLlama3` nor either
`kQwen2` variant. Mapping it onto an existing pattern would silently mistokenize,
which is exactly the failure mode the existing `qwen2`-vs-`qwen35` comment in
that function refuses. The honest fix is a new `SplitPattern` for the GPT-4o
family, with its own test, and it is **OWED** --
issue [#347](https://github.com/mudler/vllm.cpp/issues/347).

**Consequence for this row: a quant-matched llama.cpp comparison is not possible
today.** Ours runs bf16 (~56 GB on disk) and llama.cpp ships k-quants only, so
the only common artifact is the GGUF — and we cannot open its tokenizer. Running
our bf16 against a 4-bit GGUF would report a quantization difference as a speed
difference, so that cell is recorded **not comparable** rather than published.

### 11.2 The llama.cpp datapoint, and why it is not a result

Built from `ggml-org/llama.cpp` master `030ebb5` (Muse support merged in PR
[#26841](https://github.com/ggml-org/llama.cpp/pull/26841), 2026-08-10; the local
checkout at `237ad9b96` predates it), CPU-only Release, `-DGGML_NATIVE=ON`.
File `muse-glimmer-30B-kquant-17gb.gguf` (16,756,681,056 bytes, rev `2fb01e4e6f`),
which llama.cpp reports as `muse-glimmer 30B Q4_K - Medium`, 15.59 GiB, 27.85 B
params. Host: 20-core x86-64, 84 GB RAM, **no GPU**; model read over a CIFS
`soft` mount at ~117 MB/s, page-cache warmed before the run.

```
llama-bench -m muse-glimmer-30B-kquant-17gb.gguf -p 32 -n 8 -r 1 -t 4
  pp32  9.79 t/s      tg8  0.79 t/s
```

**Why this is not a result.** One repetition, so there is no noise band at all;
4 threads of 20; and the box was carrying four other agents' concurrent builds
and test suites at load average 39 to 123 with the root filesystem repeatedly at
100%. `.agents/benchmarking.md` requires the band to be calibrated from repeated
identical legs *before* a delta is interpreted, and requires reproduction on an
idle box. Neither was possible. A `-p 512 -n 64 -r 3 -t 20` leg was attempted
and abandoned unfinished for the same reason. The number is recorded so the next
attempt has something to disagree with, and for no other purpose.

### 11.3 What is owed

1. ~~The GPT-4o-family `SplitPattern`~~ — LANDED, §12.
2. **The GGUF forward** ([#359](https://github.com/mudler/vllm.cpp/issues/359),
   §12.1): coherent generation from the k-quant — ~~OWED~~ **DONE, §13** — and
   then token-exactness against llama.cpp or the bf16 arm, which is STILL OWED
   (§13.4). Until that lands the quant-matched comparison #333 wants has no
   accepted correctness gate behind it.
3. Re-run both CPU legs on an idle box, `-r` >= 3, threads matched, band first.
4. The bf16 pair (HF `exportable-muse` vs ours) on an idle dgx holding
   `${GPU_LOCK}`, one large model resident at a time.
5. The vLLM axis stays open until vllm#51655 merges and the pin advances.

## 12. The GPT-4o pre-tokenizer landed, and what it uncovered (2026-08-11, `row/TOK-LLAMA4-PRE`, issue [#347](https://github.com/mudler/vllm.cpp/issues/347))

§11.1's blocker is closed. `SplitPattern::kGpt4o` implements the GPT-4o / o200k
split; `Tokenizer::FromGguf` accepts the four pre names llama.cpp maps to
`LLAMA_VOCAB_PRE_TYPE_GPT4O` (`gpt-4o`, `llama4`, `kanana2`, `talkie`,
llama.cpp `src/llama-vocab.cpp:2294-2299` @ `153d324bcf`).

**Ported the ORIGINAL regex, not llama.cpp's transcription of it.** llama.cpp
cannot spell `\p{Lu}`/`\p{Ll}` in its engine and approximates the two letter
classes with `((?=[\p{L}])([^a-z]))` / `((?=[\p{L}])([^A-Z]))`, which drops
`\p{M}` and puts every non-ASCII cased letter in both. We implement the string
llama.cpp itself cites as authoritative (`llama-vocab.cpp:432`), which is
byte-identical to the checkpoint's own
`tokenizer.json` `pre_tokenizer.pretokenizers[0].pattern.Regex`. MEASURED on the
57-entry corpus through the 16.76 GB k-quant: **ours 0 mismatches vs HF
`tokenizers`; llama.cpp `153d324bcf` 4 mismatches** (Devanagari, Thai and
Arabic, exactly where its approximation loses `\p{M}`).

**A second, silent instance of the same bug.** `DetectPattern` in
`tokenizer.cpp` selected `kLlama3` for ANY regex containing `\p{N}{1,3}`. The
GPT-4o regex contains it, so `Tokenizer::FromHfJson` on Muse Glimmer's own
`tokenizer.json` was already picking `kLlama3` — the bf16 safetensors arm has
been mistokenizing since the model landed, with no error. Now matched exactly,
before the heuristic.

**`ignore_merges` is a non-issue for this vocab, measured not assumed.** HF sets
it; GGUF cannot express it. Encoding 29704 distinct vocab-derived strings plus a
320-string mixed corpus with the flag on and off gives byte-identical ids in
every case.

### 12.1 The GGUF arm still does not generate — the blocker MOVED to the forward

With the tokenizer fixed the 16.76 GB k-quant runs end to end and produces
tokens, but they are degenerate:

```
vllm-cli --model muse-glimmer-30B-kquant-17gb.gguf \
         --prompt "The capital of France is" --max-tokens 12 --temperature 0
  -> prompt_tokens=5   " is is is is is is is is is is is is"
```

Isolated three ways, all on this 20-core CPU-only box:

* **the file is fine** — llama.cpp `153d324bcf` `llama-completion` on the SAME
  GGUF, same prompt, `--temp 0`: `"Paris. It is the most populous city in France
  and"`;
* **the tokenizer is fine** — `prompt_tokens=5` is the correct GPT-4o
  tokenization, and the corpus diff above is 0/57 against HF;
* **the load is fine** — `test_muse_glimmer_gguf` is 12/12 with `VLLM_MUSE_GGUF`
  (428 assertions) and 12/12 / 1064 assertions with `VLLM_MUSE_GGUF_LOAD`, i.e.
  the whole 16.76 GB materialised;
* **the CPU backend is fine in general** — `vllm-cli` on `opt-125m-bf16-st`,
  same box and build: `" the capital of the French Republic."`.

So the defect is in the Muse Glimmer forward over GGUF weights (numerics or
wiring), not in the tokenizer, the loader accounting, or the backend. §10.6's
"no forward was run on GGUF weights" is now "a forward runs and is wrong", which
is a different and newly actionable claim. It is NOT fixed here: it gets its own
issue rather than a silent fix folded into a tokenizer change:
[#359](https://github.com/mudler/vllm.cpp/issues/359).

**No speed axis is claimed from any run in this section.** Every number above is
a correctness observation on a CPU-only box; the throughput cells in §11 are
untouched.

## 13. The GGUF forward defect, found and fixed (2026-08-11, `row/MUSE-GGUF-FORWARD`, issue [#359](https://github.com/mudler/vllm.cpp/issues/359))

§12.1's defect was a FOURTH convert-time transform the loader did not know about:
**`attn_q` and `attn_k` are stored in ggml's interleaved-RoPE row order.**

### 13.1 How it was found — bisect against the bf16 checkpoint, not against a theory

The three transforms §10.1 documents were the obvious suspects and all three were
INNOCENT. Rather than reason about them, every layer-0 tensor of the released
k-quant was dequantized and diffed element-wise against
`meta-models/Muse-Glimmer-30B`:

| GGUF tensor | encoding | mean rel err vs safetensors |
|---|---|---|
| `blk.0.attn_v` | Q6_K | 0.029 |
| `blk.0.ffn_down` | Q6_K | 0.028 |
| `blk.0.attn_output` | Q4_K | 0.079 |
| `blk.0.attn_gate` | Q4_K | 0.076 |
| `blk.0.ffn_gate` / `ffn_up` | Q4_K | 0.077 |
| `token_embd` | Q4_K | 0.075 |
| `output` | Q5_K | 0.052 |
| the four sandwich norms, un-shifted by 1 | F32 | **0** (exact) |
| `output_norm`, NOT un-shifted | F32 | **0** (exact) |
| `blk.0.attn_q` | Q4_K | **1.398** |
| `blk.0.attn_k` | Q4_K | **1.410** |

Every tensor sits at its encoding's quantization noise except `attn_q` and
`attn_k`, which sit at ~1.4 — the error of comparing unrelated numbers. Reading
them through llama.cpp's `LlamaModel.permute` puts them back at 0.077, i.e.
exactly the Q4_K noise of their siblings. Confirmed at layers 0, 3, 25 and 51,
with `n_head` = 32 on the query side and `n_head_kv` = 2 on the key side. The
same table CONFIRMS the three suspected transforms were correct all along.

### 13.2 The root cause

llama.cpp's converter applies, per head,
`w.reshape(heads, 2, Dh/2, K).swapaxes(1, 2)` to the query and key projections
(`conversion/llama.py:163-169`, inherited by the `muse-glimmer` converter of
ggml-org/llama.cpp#26841). That is the weight-side half of ggml's `rope_norm`,
which rotates ADJACENT channel pairs `(2i, 2i+1)`; HF — and our `vt::RopeNeox` —
rotates HALF-OFFSET pairs `(i, i + Dh/2)`. Consuming the file's rows verbatim
therefore rotated the wrong channel pairs on the 39 RoPE layers.

**Why every existing gate passed.** The permutation preserves names, shapes,
counts, dtypes and the tensor accounting, so §10.6's 731/731 structural result
was true and irrelevant. It is also invisible to attention on the 13 NoPE layers:
a permutation applied to BOTH q and k leaves `q·k` unchanged. Nothing short of an
element-wise comparison against the bf16 checkpoint, or a forward with RoPE on,
could see it — which is why the model loaded, ran, and emitted `" is is is ..."`
instead of failing.

### 13.3 The fix and its evidence

`LoadMerged` in `muse_glimmer_gguf_weights.cpp` now carries a per-shard head
count and un-permutes the q and k rows on the way into the merged `qkv_proj`, on
BOTH residency paths — the dequantized one the released heterogeneous trio takes,
and the kept-quant byte concat a homogeneous trio would take (whole rows are
whole numbers of blocks, so the reorder requantizes nothing). `attn_v`,
`attn_output` and the MLP shards are untouched.

Generated text on the released 16.76 GB k-quant, same box, same build, same
prompt, `--temperature 0`:

```
"The capital of France is"             (12 tokens, --temperature 0)
  before: " is is is is is is is is is is is is"
  after:  " Paris. The capital of France is Paris. The capital of"

"The history of the Roman Empire began" (20 tokens, --temperature 0)
  after:  " with the founding of the city of Rome in 753 BC and ended in
           the west with the fall"
```

The second prompt is there because the first one's after-text loops, and a loop
is weak evidence on its own. It does not loop, and it is factually right.

Gate `tests/vllm/models/test_muse_glimmer_gguf.cpp` grew two cases — one per
residency path, the kept-quant one on a new Q8_0 synthetic file whose geometry
lets a homogeneous trio exist at all. RED first with the loader reverted: 2 cases
/ 140 assertions failing, the other 12 cases green. GREEN with the fix: 14/14,
640 assertions; 706 with `VLLM_MUSE_GGUF` on the released file and 1342 with
`VLLM_MUSE_GGUF_LOAD` (the whole 16.76 GB materialized).

Five mutations, each proven RED and the tree restored byte-for-byte (md5
`3fc4335f`):

| Mutation | Result |
|---|---|
| un-permute `attn_k` with `n_head` instead of `n_head_kv` | 2 cases / 32 assertions RED |
| also un-permute `attn_v` | 2 cases / 46 assertions RED |
| invert the map the other way (`(i%2)*half + i/2`) | 1 case / 90 assertions RED |
| skip the un-permute on the KEPT-QUANT path only | 1 case / 91 assertions RED |
| skip the un-permute on the DEQUANTIZED path only | 1 case / 49 assertions RED |

Adjacent gates unchanged and green on the same build: `test_muse_glimmer_text`
21/21, `_wiring` 9/9 (10316 assertions), `_scaffold` 11/11, `_vision` 7/7,
`_real_weights` 3/3, `test_gguf` 33/33, `test_qwen3_dflash_gguf` 3/3.

### 13.4 What is STILL not established

- **Not token-exact against anything.** llama.cpp `153d324bcf` on the same file
  and prompt gives `"Paris. It is the most populous city in France and"`; ours
  agrees on the first token and then diverges into a repetition. Whether that
  residual is quantization/accumulation drift, the `post_norm_eps` fallback of
  §10.3, or a second defect is OPEN and owed. "Coherent" is the claim;
  "token-identical" is not. **Updated 2026-08-11:** the `post_norm_eps` candidate
  is ELIMINATED — §15 fixes the fallback and measures the generated tokens on both
  arms; the divergence from llama.cpp is unchanged by it.
- **No speed axis, still.** §0 is unchanged: the pinned oracle cannot load
  `muse_glimmer`, so there is no denominator and none is claimed here.
- The bf16 safetensors arm at full depth has still never generated; it is
  unaffected by this fix, which is GGUF-only. (Superseded by §14: it generated
  4 ungated tokens on 2026-08-11.)

## 14. The speed re-run — first binding numbers on any axis (2026-08-11, `row/MUSE-BENCH-2`, issue [#333](https://github.com/mudler/vllm.cpp/issues/333))

§11 produced no binding number because our GGUF arm could not reach a forward.
`11d45330` (the `llama4`/GPT-4o pre-tokenizer, #347) and `75a29016` (the Q/K
interleaved-RoPE row order, #359) removed both blockers, so for the first time
**both engines can hold AND generate from the same file**, which is what makes a
quant-matched comparison possible at all.

**vLLM remains an OPEN GAP by construction.** §0 is unchanged: the pin carries
no `muse_glimmer`, so there is no denominator, nothing was substituted for it,
and nothing below is a parity claim. llama.cpp is a **labelled SECONDARY**
reference per [[vllm-is-the-bar-not-llamacpp]].

### 14.1 The one thing that makes these numbers comparable

Every llama.cpp cell reads **the same 16,756,681,056-byte file** we read
(`muse-glimmer-30B-kquant-17gb.gguf`, md5 `ba8da9b15aed63a1df095cb34f3e7665`,
copied to local NVMe so neither engine pays a CIFS tax inside a measured
window). Every HF cell is bf16 safetensors against **our** bf16 safetensors.
No cell mixes a 4-bit file against a 16-bit one, because that reports
quantization as speed.

Host `dgx.casa`, GB10 aarch64, 20 cores (10 Cortex-X925 + 10 A725), **idle**
(load 0.23 at claim, `local-ai-worker` already down, GPU 0%), one `$HOME/gpu.lock`
held across the whole series, legs **paired and order-alternated**.

### 14.2 GGUF k-quant, CPU, quant-matched — the binding cells

Ours = `vllm-bench` `Prefill token throughput (in/TTFT)` and
`Mean per-stream decode rate`; llama.cpp = `llama-bench` `pp`/`tg`. The two
harnesses measure the same two quantities and their absolute scales agree, so
[[conflicting-ratios-check-absolute-scale-first]] does not bite here.

| Workload | Axis | vllm.cpp | llama.cpp | Ratio |
|---|---|---:|---:|---:|
| in 128 / out 16, t=20, r=5 | prefill tok/s | 11.63 (10.36-12.10, **15.0%**) | 12.94 (12.90-13.09, 1.4%) | **0.898x** |
| in 128 / out 16, t=20, r=5 | decode tok/s | 1.18 (1.04-1.71, **56.8%**) | 5.08 (4.88-5.28, 7.9%) | **0.232x** |
| in 128 / out 16, t=10, r=3 | prefill tok/s | 9.94 (1.2%) | 9.97 (0.8%) | **0.997x** |
| in 128 / out 16, t=10, r=3 | decode tok/s | 1.31 (0.8%) | 6.41 (0.7%) | **0.204x** |
| in 512 / out 16, t=20, r=3 | prefill tok/s | 2.23 (0.4%) | 13.13 (0.5%) | **0.170x** |
| in 512 / out 16, t=20, r=3 | decode tok/s | 0.29 (0.0%) | 5.00 (9.6%) | **0.058x** |
| any | peak RSS | 30.29 GiB | 15.74 GiB | **1.92x MORE** |

Values are medians; brackets are min-max and the spread as a percentage of the
median. The noise band was calibrated from the repeated identical legs
themselves, per `.agents/benchmarking.md`; no leg was discarded in this table.

### 14.3 What the shape of the table says

**llama.cpp is flat in context length and we are not.** Its prefill goes
12.94 -> 13.13 tok/s from 128 to 512 input tokens (+1.5%); ours goes
11.63 -> 2.23 (**-81%**). Its decode is 5.08 -> 5.00 (flat); ours is
1.18 -> 0.29 (**-75%** for 4x the context). At 128 tokens we are within 10% of
it on prefill and tie it outright at 10 threads; the gap is created by
sequence length, not by the GEMMs.

**Idle cores are NOT the explanation; that hypothesis is REFUTED.** The
aggregate `Percent of CPU` of a whole run reads 466-667% for us against
1062-1674% for llama.cpp, which looks like an idle-core story and is not one:
our run is dominated by a load-and-dequantize phase that llama.cpp's mmap never
pays. A two-length diff (out 4 -> out 36, in 128, the common load phase
cancelling) gives the decode phase itself:

| Engine | dCPU-seconds | dwindow | decode-phase CPU | CPU-s per decoded token |
|---|---:|---:|---:|---:|
| vllm.cpp | 599.95 | 30.19 s | **1987%** (19.9 of 20 cores) | **18.75** |
| llama.cpp | 106.26 | 5.87 s | **1810%** (18.1 cores) | **3.32** |

Both saturate the box, and we burn **5.6x the CPU-seconds per decoded token**.
Read that as resource cost, **not** as proof the extra seconds are useful work:
the 2026-08-06 CPU op-dispatch profile in the benchmark record already measured
decode on this backend as **47.15% threadpool synchronisation** (`ThreadReady` +
`PollForWork` + `Barrier`; ~58% on the secondary-thread view), and a spinning
worker holds a core without advancing a token. The refuted claim is "our pool
leaves cores idle"; the live one is "our pool burns them at the barrier", which
is that already-named lever.

**Neither lever is new; this row prices them on a second model.** `.agents/NOW.md`
carries both from that profile: decode 47% threadpool sync, and prefill ~39% CPU
paged attention of which ~21% of prefill sits in a **per-element dtype switch in
the attention dot loop** (`src/vt/cpu/cpu_paged_attn.cpp:29`, called from `:143`),
the same defect class E1 hoisted out of the elementwise GEMM. What is new here is
the **context-length shape**: llama.cpp is flat from 128 to 512 input tokens
while our prefill falls 81% and our decode 75%. That is what an attention inner
loop paying per element per key looks like beside one that does not, and it
matches [[no-fa2-arch-means-fallback-attn-is-the-wall]]. The owed next step is a
decode-window profile of our CPU attention kernel against `ggml`'s on this
workload. **No ceiling is claimed.**

One observation IS new: at 20 threads our decode spread is **56.8%** against
llama.cpp's 7.9%, and at 10 threads ours collapses to **0.8%**. GB10's 20 cores
are 10 Cortex-X925 plus 10 A725, and our barrier is unstable across that split
in a way llama.cpp's is not, which is consistent with the 47% synchronisation
finding and is cheap to test.

**Memory is an open gap.** 30.29 GiB resident against llama.cpp's 15.74 GiB for
the same 15.59 GiB file, exactly as §10.2 predicts: we dequantize `qkv_proj`,
`lm_head` and `embed_tokens` while llama.cpp keeps every operand in blocks.

### 14.4 Correctness context, captured on the same binaries

Speed without its correctness state is not a result. On the identical build and
file, prompt `"The capital of France is"`, `--temperature 0`:

| Engine | 12 greedy tokens |
|---|---|
| vllm.cpp (GGUF) | `" Paris. The capital of France is Paris. The capital of"` |
| llama.cpp (GGUF) | `"Paris. It is the most populous city in France and"` |
| vllm.cpp (bf16) | `" Paris. It is"` (4 tokens) |

Coherent, **not token-identical**; §13.4's residual is unchanged and unclosed.

The HF bf16 reference was probed on CPU and is reachable but needs a warmup:
its FIRST 8-token forward costs **660.47 s** of one-time lazy materialization and
the subsequent `generate` costs **5.74 s**, so any HF leg timed without a warmup
measures page-in, not compute. Load 90.94 s, VmHWM 55,914,364 kB. Our bf16 arm's
peak RSS is 59,819,588 kB.
One thing is new: **the bf16 safetensors arm generated at full depth for the
first time** (§13.4 recorded that it never had), and its first three tokens
`" Paris. It is"` agree with **llama.cpp's** continuation rather than with our
own GGUF arm's. That is a directional hint that the §13.4 divergence lives in the
GGUF path and not in the model wiring, and it is a hint, not a finding: three
tokens are not a gate. An owed follow-up is a GGUF-vs-bf16 token diff at depth.

### 14.5 Blocked cells, each with its named blocker

| Cell | Blocker |
|---|---|
| **vLLM, any arm** | **OPEN GAP by construction.** The pin has no `muse_glimmer`; unblocks when vllm#51655 merges and the pin advances. Not waived, not substituted. |
| ours bf16 on GPU vs HF bf16 on GPU | **Disk.** A CUDA build tree is ~169 GiB (`.agents/benchmarking.md`); dgx had ~122 GiB free. A full disk voids a binding silently, so no CUDA build was started. |
| ours bf16 vs HF bf16, CPU, r=3 | **Host availability, twice.** The series was written, syntax-checked and queued on `$HOME/gpu.lock`, then stopped rather than run contended: first behind a foreign CUDA build (14:26, `ptxas`/`cicc` at 100%), then behind another session's `vllm-server` serving gate holding the lock from 15:00. The lock was respected both times. **OWED**, scripts staged on dgx. |
| clean r=3 repetitions of the two-length CPU diff | Same queue. The r=1 legs in §14.3 are clean and pre-date the foreign build; only the repetitions are owed. |
| mmproj / perception encoder | Still refused by name (§10.4): `v.patch_embd.weight` lacks the `patch_temporal` axis. No multimodal speed axis exists. |
| DFlash draft acceptance A/B | Structurally reachable (§10.5), never exercised. |

### 14.6 What these numbers do and do not establish

They establish that on one idle 20-core aarch64 host, reading one identical
k-quant file, vllm.cpp is **at parity on short-context prefill** (0.997x at 10
threads, 0.898x at 20), **~4-5x behind on decode**, **~5.9x behind on 512-token
prefill**, and **uses 1.92x the resident memory** — with the decode gap
attributed to CPU work per token rather than to idle cores, and the
context-length gap pointing at the attention path.

They establish **nothing about vLLM**, which is the only bar that counts and
remains unavailable; nothing about GPU behaviour, which was not built; nothing
about multimodal; and nothing about token-exactness. **No ceiling is claimed or
implied anywhere in this section.**

## 15. Config defaults are the ARCHITECTURE's, not neutral values (2026-08-11, `row/FIX-MUSE-CONFIG-DEFAULTS`, issue [#412](https://github.com/mudler/vllm.cpp/issues/412))

Found by a differential review of SGLang's independent implementation
(`muse-glimmer` @ `38a1bc5d2f`) against ours. Same shape as
[#405](https://github.com/mudler/vllm.cpp/issues/405), generalized: that fix
corrected one field, this is the class.

When a key is absent we fell back to a **neutral** value (identity, zero, "off").
Both references fall back to the **architecture's constant**. For a key the
released 30B `config.json` always ships the two agree, which is exactly why the
defaults went untested — the only checkpoint that can reach a default is one
that OMITS the key, and two such checkpoints exist.

### 15.1 The six fields, and both references' authority for each

| key absent | ours (before) | vLLM #51655 @ `075d645af` | SGLang @ `38a1bc5d2f` | now |
|---|---|---|---|---|
| `qk_scale_factor` | `scale_query_by = 1.0` | `configs/muse_glimmer.py:62` 43.7840518911 | `srt/configs/muse_glimmer.py:98` same | 43.7840518911, folded by `sqrt(head_dim)` → exactly 3.87 at 128 |
| `sliding_window` | **0 = no window at all** | `:56` 2048 | `:93` 2048 | 2048 |
| `output_multiplier` | 1.0 | `:65` 0.19611613513818404 | `:101` same | 0.19611613513818404 |
| `final_logit_softcapping` | 0 = none | `:58` 20.0 | `:102` `output_soft_cap_temp` 20.0 | 20.0 |
| `rms_norm_eps` | 1e-6 | `:45` 1e-5 | `:90` 1e-5 | 1e-5 |
| `post_norm_eps` | `rms_norm_eps` | `:67` 1e-8 | `:91` 1e-8 | 1e-8 |

The two references AGREE on all six, so nothing was adjudicated and no
`NEEDS_DECISION` arises. They are named constants in
`include/vllm/model_executor/models/muse_glimmer.h` so the parser, the GGUF arm
and the struct's own member defaults cannot drift apart — the struct was a
SECOND copy of the same trap.

`sliding_window` and `final_logit_softcapping` are `X | None` upstream, where
`None` is a real state distinct from absent. Absent now takes the default; an
explicit `"sliding_window": null` still means no window, in both the nested and
the flat layout (the flat hoist used to drop a null and re-default it).

### 15.2 It was LIVE on the GGUF path

The released `muse-glimmer-30B-kquant-17gb.gguf` has 32 KV pairs; its header was
dumped and it carries **no** `attention.post_norm_rms_epsilon` and **no**
`attention.scale`. So both sandwich post-norms ran at
`attention.layer_norm_rms_epsilon` = 1e-5 where the released safetensors
`config.json` says 1e-8 — a factor of 1000, on the only arm of this model most
people can run. §10.3's "not observable" argument assumed a post-norm input mean
square of order 1; the sensitivity goes as `1/ms` and nothing measured it.

The arm now also PREFERS what a file says over any default: a converter emitting
`muse-glimmer.attention.post_norm_rms_epsilon` is honoured, and one emitting
llama.cpp's `%s.attention.scale` (`LLM_KV_ATTENTION_SCALE`, `llama-arch.cpp:240`
— the factor that replaces `head_dim ** -0.5` in the softmax, so
`scale_query_by = scale * sqrt(head_dim)`) wins over the value folded into
`attn_q_norm`. `attention.sliding_window`, `logit_scale` and
`final_logit_softcapping` are likewise written only when the file carries them,
so exactly ONE place decides a default.

### 15.3 It was latent via the DFlash drafter

`meta-models/Muse-Glimmer-30B-Assistant`'s `config.json` omits
`qk_scale_factor`, `post_norm_eps`, `output_multiplier`,
`final_logit_softcapping` and `vocab_size`. Routed through
`ParseMuseGlimmerParams` it ran with `use_qk_norm = true` and **no query
pre-scale at all** — coherent drafts, no error, acceptance quietly collapsing.
That is the silent-degradation signature §1.3 already warns about for a
draft/target RoPE mismatch. The file is now committed verbatim as
`tests/vllm/models/fixtures/muse_glimmer_30b_assistant/config.json` and is what
pins those four defaults.

**No claim is made that the drafter now works.** Nothing here ran it. A trap was
removed; the acceptance-rate A/B of §10.5 is still OWED.

### 15.4 Evidence

RED first. The critical shape of #405 was that its test set the flag
EXPLICITLY, which is why the default stayed untested — so every new assertion
here is made against a config that OMITS the field.

| Gate | RED (before the fix) | GREEN |
|---|---|---|
| `test_muse_glimmer_scaffold` | 1 case / 3 assertions failing of 85 | 11/11, 85 assertions |
| `test_muse_glimmer_text` | 3 cases / 16 assertions failing of 528 | 24/24, 528 assertions |
| `test_muse_glimmer_gguf` | 3 cases / 10 assertions failing of 654 | 17/17, 654; 727 with `VLLM_MUSE_GGUF` on the released 16.76 GB file |

A trap inside the RED capture is worth recording: `doctest::Approx` carries a
`scale` term defaulting to 1.0, so a bare `CHECK(eps == Approx(1e-8))` accepts
1e-5 — the exact wrong value — as equal. Three assertions passed against the
defect until every epsilon comparison got `.scale(0)`. The pre-existing released-
config assertion had the same hole.

Eleven mutations, each reverting one default (or one precedence rule) to what it
replaced, each proven RED, tree restored byte-for-byte from an in-memory
snapshot and md5-verified:

| Mutation | Caught by |
|---|---|
| `rms_norm_eps` default 1e-5 → 1e-6 | text 1, gguf 1 |
| `post_norm_eps` default 1e-8 → 1e-5 | text 3, gguf 3 |
| `sliding_window` default 2048 → 0 | text 3, gguf 1 |
| `output_multiplier` default → 1.0 | text 2, gguf 1 |
| `final_logit_softcapping` default → 0.0 | text 4, gguf 1 |
| absent `qk_scale_factor` → identity | scaffold 3, text 3 |
| an explicit `null` no longer disables the window / cap | text 2 |
| the flat hoist drops an explicit `null` again | text 1 |
| GGUF ignores `attention.post_norm_rms_epsilon` when present | gguf 2 |
| GGUF ignores `attention.scale` when present | gguf 2 |
| GGUF writes the three optional keys unconditionally | gguf 3 |

Adjacent gates unchanged and green on the same build: `_wiring` 9/9 (10317),
`_vision` 7/7 (98), `_real_weights` 3/3 (15), `test_model_registry` 24/24 (871),
`test_tool_parser_muse_glimmer` 15/15, `test_reasoning_muse_glimmer` 19/19, and
`test_muse_glimmer_text` again under `VT_FUSED_CHAIN_ADOPT=0` (24/24, 528).

**The text tower did not move**, as predicted: the released `config.json` carries
all six keys, so the fp32-reference comparisons of §6.5/§6.6 are untouched. Only
a checkpoint that omits a key changes, and the safetensors 30B omits none.

### 15.5 Did GGUF numerics move? MEASURED, and the answer is "not in the tokens"

Same-binary A/B on the released 16.76 GB k-quant, this box, CPU-only Release,
`--temperature 0`, `flock /tmp/cpu-bench.lock`. The ONLY difference between the
two arms is `kMuseGlimmerDefaultPostNormEps`; both prompts are the two §13.3
recorded.

| prompt | 1e-8 (fixed) | 1e-5 (the old fallback) |
|---|---|---|
| `"The capital of France is"`, 12 tok | `" Paris. The capital of France is Paris. The capital of"` | **identical** |
| `"The history of the Roman Empire began"`, 20 tok | `" with the founding of the city of Rome in 753 BC and ended in the west with the fall"` | **identical** |

Both arms also reproduce §13.3's recorded text exactly, so this is a three-way
agreement, not two runs of one build.

**What that does and does not say.** The generated tokens do not move at these
lengths — the change is not large enough to flip a greedy argmax here. It does
NOT say the activations are unchanged: the norm's scale factor changes by
`1 - sqrt((ms + 1e-8)/(ms + 1e-5))`, which is `5.0e-6` at a post-norm input mean
square of 1 and grows as `1/ms`. It crosses half a bf16 ulp (`2^-9`) at
`ms = 2.6e-3`, i.e. a post-norm input **RMS of 0.051**: below that the difference
is representable in the activation dtype and rises fast (4.6e-2 at RMS 0.01).

So §10.3's conclusion happens to hold for THESE two prompts, while its
*reasoning* — the assumed `ms` of order 1 — is still unmeasured, and the
threshold above is what would have to be measured to justify it. That is
precisely why the fix does not depend on it: 1e-8 is the architecture's value in
both references, so the correct fallback costs nothing and needs no argument
about magnitudes. **Two greedy prompts are not a token-exactness claim.**

### 15.6 What is still open

- **Still not token-exact against llama.cpp.** §13.4 listed the `post_norm_eps`
  fallback as one of three candidate explanations for the residual divergence.
  That candidate is now eliminated; quantization/accumulation drift and a
  possible second defect remain, and the comparison is still OWED.
- **The drafter has still never run.** §10.5's acceptance A/B is unchanged.
- **No speed axis.** §0 stands: the pinned oracle cannot load `muse_glimmer`, so
  there is no denominator and none is claimed here on any axis.

## Owed

- [#1466](https://github.com/mudler/vllm.cpp/issues/1466) —
  `tests/vllm/models/test_muse_glimmer_text.cpp:532`, `CHECK(diff <= 5e-4)` is
  the W1 measurement rounded up (`3a54c4b7d`'s body quotes 1.21e-04) with no
  derivation beside it. `4712dac40` gave `vt`'s gated activations upstream's
  rounding polarity, which is correct, and grew the envelope 2.8x to 3.43e-04 —
  0.687 of the bound. Found while gating
  [#1458](https://github.com/mudler/vllm.cpp/issues/1458), which repairs the
  other member of the same class in the same case (`bdiff <= 1e-5`) and
  deliberately leaves this one to its own derivation. A repair owes a measured
  floor or a precision argument, never a bigger constant.

- [#1566](https://github.com/mudler/vllm.cpp/issues/1566) — **the perception
  encoder is an UNREACHED staged slice, so everything W3 built is dead code
  today.** `MuseGlimmerVisionForward` is reached only from
  `MuseGlimmerEncodePixelGroups` (`muse_glimmer_mm.cpp:191`) and
  `MuseGlimmerGenerateGreedyViaRegistry` (`:276`), and neither has a caller in
  `src/`, in `examples/` or in `include/vllm.h` — only in `tests/`.
  `muse_glimmer_registry.cpp:13-14` states it: *"The perception encoder is still
  W3, so an image or video prompt is a pending brick."* §3's W4 and W5 own the
  wiring. Until they land, every change inside the tower is the staged-slice
  exception in `.agents/reachability.md` and has to name this issue in its commit
  body and its pull request body. The first one that does is
  [#1545](https://github.com/mudler/vllm.cpp/issues/1545), which routes the
  tower's 50 attention layers off the correctness-grade kernel
  ([muse-glimmer-vision-attn-flash.md](muse-glimmer-vision-attn-flash.md)). The
  issue exists because the model's umbrella
  [#268](https://github.com/mudler/vllm.cpp/issues/268) is closed, so there was
  nothing open left to name.
