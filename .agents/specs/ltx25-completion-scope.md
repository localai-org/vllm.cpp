# SPEC — `LTX25-COMPLETION-SCOPE`: what it costs to call LTX-2.5 complete

Issue: [#2526](https://github.com/mudler/vllm.cpp/issues/2526).
Owner row: `LTX25-COMPLETION-SCOPE`.
Base: `origin/main` at `63889449c`. Upstream oracle: Lightricks `LTX-2` at `fd4ded7f`.

**Every count and every anchor in this document was re-derived from the tree
at `781ea5487` after a fresh review returned NOT PASS.** The review's diagnosis
is recorded here because it shapes what a later reader should distrust: the
method held wherever this row read UPSTREAM and failed wherever it read OUR OWN
TREE. Two of its wrong anchors had been inherited from `ltx-2-5.md` rather than
re-derived, in the section whose own thesis is that a record was propagated
rather than read. Two class-A entries described work that had already shipped,
one described something upstream never built, and one pointed at a refusal for a
different capability entirely. Nothing about this tree is quoted from another
document below, and **every count is now produced by a command in `## Gates`
that prints its own denominator** — a number this row cannot re-derive on demand
has been withdrawn rather than restated.

**A second review then falsified three of this document's own closures, and the
diagnosis is the same one a level up.** The counts held; the POPULATIONS did not.
A capability gap in no class at all — the phase-L6 bf16 arms of eight non-DiT
components, now **A24** (§A.7) — was invisible to every sweep here because seven
of its eight members refuse nothing, and a survey that enumerates refusals cannot
see a component that simply computes in the wrong dtype. D11 was published as one
records defect and is **nine** (§6.1). Class B was published as four correct
mirrors and is **eight** (§3). Each was a closure claimed over a population
nobody had swept with a stated rule, which is exactly what §2.1 says about counts
and now has to be said about populations too.

**Every figure in `## Gates` was re-run at `337130aa2` during that repair and
each reproduced unchanged** — 52 files, 375/61/270/53, 98 case-insensitive
against 64 case-sensitive, 10 kinds and 28 pairs, cadence 43, and the Python
block's 53/49/57/352, 57 invisible, 166/159/7. The `781ea5487` stamp above is
kept because it is where the figures were first derived; a second tree agreeing
with it is evidence, not a reason to restamp.

## Scope

**In:** an enumeration of what stands between `63889449c` and a complete LTX-2.5
port, with every item classified, anchored on both sides, and sized; the
sequencing of the capability gaps; and one honest total.

**Out:** implementing any arm. This row is a survey and it returns
`NEEDS_DECISION` on what to fund. It writes no product code.

Every count below is stamped with the SHA it was taken at, because a count with
no tree beside it cannot be reconciled. Rerun the commands in `## Gates`.

## 1. The distinction this row exists to draw

A grep over the LTX-2.5 sources returns 375 `Fail()` sites — `ltx2*.cpp` and
`ltx2*.h` under `src/vllm/model_executor/models`, `src/vllm/multimodal`,
`include/vllm/model_executor/models` and `include/vllm/multimodal`, the file set
`## Gates` names. They are not one population. Three kinds share the text:

1. **Input validation.** `"Video width and height must be multiples of 32"`,
   `"the checkpoint is missing '...'"`. The overwhelming majority. Not debt.
2. **A refusal that MIRRORS upstream.** We refuse because upstream refuses, or
   because upstream has no such thing to run. AGENTS.md requires this: mirror
   "every applicable mode, default, error, and edge case". Removing one would be
   a parity DEFECT, not progress.
3. **A refusal that says we did not port it.** This is the work.

Kinds 2 and 3 are textually identical. `git grep "not ported"` returns both and
distinguishes neither. This row separates them by reading the upstream side of
each at `fd4ded7f` and asking one question: **does upstream construct this?**

**`Fail()` is not the whole refusal surface, and this row's first draft assumed
it was.** The same file set carries **61 `Refuse(`, 270 `VT_CHECK(` and 53
`Require(`** at `781ea5487`, and **five of this document's own anchors live in
the three excluded populations**: A7 (`ltx2_video_vae.cpp:1355`), E1
(`ltx2_device.cpp:1337`) and `ltx2_dit.cpp:766` are `VT_CHECK`; A8 reaches
`Refuse(` through `Ltx2RefuseUnportedPipelineFeature`; A9
(`ltx2_upsampler.cpp:467`) is `Require`. The 375 is therefore a lower bound on
the refusal surface and never a denominator. `## Gates` prints all four.

**The consequence was not cosmetic: the first pass enumerated only the refusals
it already knew about.** Sweeping all four populations, case-insensitively, found
nine capability gaps in no class at all. §A.5 carries the sweep and `## Gates`
carries the command.

**And the refusal surface is not the CAPABILITY surface either.** A gap can refuse
nothing and still be a gap: the code simply computes something upstream does not,
and a header comment records the debt. A24 is that shape — upstream constructs
eight components in bfloat16, this tree computes them in f32, and exactly one of
the eight refuses by name (§A.7). No count over `Fail(`, `Refuse(`, `VT_CHECK(`
and `Require(` can reach the other seven, at any width and in any case. **Class B
has the same ceiling**, and it cost four members: three guiders and one VAE block
are refused at their own call sites and registered in neither the unported-feature
enum nor the campaign's `Out` list (§3).

**The upstream-construction test is necessary, not sufficient, and one recorded
gap fails it.** It separates a mirror from a gap. It does not separate a gap from
a place where OUR host path has a capability and OUR device path refuses it —
the prompt K/V cache, which upstream never builds either. Rather than smuggle
that through the test or drop it for failing, it gets **class E** (§A.6), so the
test stays exactly as strong as it reads.

## 2. The classes

| Class | Meaning | Count |
|---|---|---:|
| **A** | Genuine capability gap: upstream implements it, we refuse or lack it | **21** |
| **B** | Correct mirror: we refuse because upstream refuses or never constructs it | **8** |
| **C** | Externally blocked: unfinishable here at any effort | **0** |
| **D** | Measurement and records debt: no engine change | **12 clusters** |
| **E** | Internal consistency gap: OUR host path has it, OUR device path refuses it | **1** |

**Class A grew from fourteen to twenty-one, and that is the honest direction.**
Two entries left it because the arms had SHIPPED before this row's own base —
found by opening the tree the first pass cited without reading (§A.4). One left
for the new class E, because it fails class A's own test (§A.6). Ten entered:
nine from a closed refusal sweep the first pass never ran (§A.5), and one — A24,
the phase-L6 dtype arms — from a second sweep, over a marker no earlier pass
swept for, which no `Fail()`/`Refuse(`/`VT_CHECK(`/`Require(` count could have
reached because most of its members are HEADER RECORDS rather than refusals
(§A.7). §"Risks" said class A was a lower bound. It was, and the reason is worth
keeping: the first pass enumerated the refusals it already knew about instead of
sweeping for them.

**Class B grew from four to eight on the same principle.** Three guider arms and
one VAE decoder block are refused here because upstream cannot construct them,
and they were in neither registry §3 read — not the `Ltx2UnportedPipelineFeature`
enum and not the campaign's `Out` list. A reader sweeping the tree therefore hit
refusals §3's mirror list did not cover, which is exactly the re-filing §3 exists
to prevent.

**Class E exists because the alternative was worse.** Class A's test is "does
upstream construct this?", and one recorded gap fails it while remaining a real
capability a user can feel. Smuggling it through the test would corrupt the test;
dropping it would lose the gap. It gets its own class, of one.

**Class C is EMPTY, and that is this row's largest single finding.** All three
recorded members were verified against `fd4ded7f` rather than inherited, and all
three are falsified. §5 carries the evidence.

### 2.1 The campaign's own `## Owed` backlog, counted properly

Classes A-D are the CAPABILITY view: what a user cannot get. The campaign also
carries an internal debt list, and it is larger than any prior count.

**352 items under an owed heading, in 57 owed sections across 49 of the 53
specs in the population**, at `781ea5487`. The number is worth nothing without
the rule that produced it, so the rule IS a command — the self-contained
`python3` block in `## Gates` prints every figure in this section — and this is
what it means, exactly:

* **Population:** `.agents/specs/ltx25-*.md` plus `.agents/specs/ltx-2-5.md`,
  minus this file.
* **Section:** any ATX heading, at any level, whose text -- after stripping an
  optional leading `N.` or `N.M.` numbering -- begins with `Owed` or with
  `What is/stays [still] owed`. The section runs to the next heading of ANY
  level.
* **Item:** a bullet at column zero, or a table data row (a `|` line that is
  neither a `|---|` separator nor immediately followed by one).

**The rule is load-bearing, and the evidence is that four rules give four
answers.** This section first published **299**. The fresh reviewer's rule gave
**362**, a second rule of the reviewer's gave **350**, a repair pass gave
**368**, and the rule stated above gives **352**. None reproduces another, and
none is wrong — they differ on whether a nested bullet counts, on which heading
wordings open a section, and on where a section ends. **A count over prose is a
property of its rule, so the rule ships with the number or the number does not
ship.** That is why 299 is withdrawn rather than corrected: nobody can say what
it counted.

**Three things this number is not.** It is not "open" items: the rule cannot
read whether an item was since discharged, and nothing in the tree marks that
mechanically. It is not 352 rows: many are one bullet of a wave and several are
duplicated across specs. And it is not a measure of remaining work, because §4
is the capability view and this is the internal-debt view.

The heaviest specs are `ltx25-device-residency` (51, over four owed sections),
`ltx25-decode-speed` (25), `ltx25-dit-attn-flash` (18), `ltx25-phase-instrument`
(15), `ltx25-t2a-one-stage` (15) and `ltx25-vae-device-residency` (13). Four
`ltx25-*` specs carry no owed heading at all.

**The ENGINE / MEASURE / RECORD split this section published is withdrawn, and
nobody can re-derive it.** It read 192 / 83 / 24. **No literal marker for those
kinds exists in the population**: `[ENGINE]`, `ENGINE:` and `kind=ENGINE` return
0, and the bare words return a handful of hits, every one of them ordinary prose
or the roadmap matrix's own unrelated category counter. A split of 299 items
into three kinds was therefore a judgement made once, by one reader, and
recorded as a measurement. It is exactly the drift-lock this repository already
names, and the number was quoted onward before anybody tried to reproduce it.
What survives is the question, and it is returned as a decision rather than
answered: **nothing in this tree records what kind of work an owed item is**, so
no gate and no plan can schedule this backlog without a human reading all 352.

### 2.2 A gate defect found while counting, and it under-reports this backlog

`owed_issues()` - the record checker's backlog reader, retired by the
local-issue-authority cutover (`a9f6186c2`) - read the backlog with two bugs:

```python
if "\n## Owed" not in text:
    continue
body = text.split("\n## Owed", 1)[1].split("\n## ", 1)[0]
```

1. **The gate recognises one literal heading and nothing else** (line 2012).
   `if "\n## Owed" not in text: continue` skips the whole file, so a numbered
   heading and a differently-worded one are both invisible. **Five LTX specs**
   are wholly invisible for this reason at `781ea5487`: `ltx25-token-append.md`
   (`## 8. Owed`), `ltx25-decode-threads.md` (`## 7. Owed`),
   `ltx25-decode-dtype.md` (`## 7. Owed`), `ltx25-text-proj-dtype.md`
   (`## 6. Owed`) and `ltx25-tiled-decode.md` (`### What is owed`, with no
   `## Owed` anywhere). The fifth was missed by this row's first pass because it
   looked for numbered headings and not for other wordings — the same shape of
   error as the defect it was reporting.
2. **Only the FIRST section is read, and it stops at the next `\n## `** (line
   2014). **The mechanism this section published was wrong, and the file it
   blamed is not a victim.** It said a `###`-level owed section is truncated
   away. `'\n## '` is NOT a substring of `'\n### '` — the third `#` sits where
   the space must be — so a `###` owed section survives, and is truncated only
   when a `## ` heading of some other name intervenes first.
   `ltx25-phase-instrument.md`'s `### Owed out of the fresh review` is INSIDE
   the gate's body: the gate reads all 15 of its items, and naming it here was a
   false accusation against a spec. The real victims of this second defect are
   **three**: `ltx25-device-residency.md` (three later owed sections lost),
   `ltx25-res2s-loop.md` (two lost) and `ltx25-text-cond-device.md` (one lost),
   plus `ltx25-anchor-repair.md`, which carries a `## Owed` the gate reads and a
   later `What stays owed` section it does not.

**57 of the 352 items sit under headings this gate cannot see** — not "roughly
74 of 299". **But item count is the wrong unit, and D10 measured a quantity the
function does not compute.** `owed_issues()` returns `set[str]` of ISSUE
NUMBERS; it never counts items at all. Measured in its own unit: **166 distinct
issue numbers appear under an owed heading, `owed_issues()` collects 159, and it
misses 7.** The blindness is real and far smaller than advertised, because most
invisible items cite an issue that some visible section cites too. Both units
are printed by the same command in `## Gates`, and the issue unit is the one a
repair should gate on, because it is the unit the function has.

`ltx25-decode-threads.md:292` says this about itself, so it was known and never
filed. AGENTS.md is explicit that this class of thing is the checker's defect,
not the specs': the obligation is real and the gate silently does not enforce
it. This row files it rather than fixing it, because a semantic checker change
needs its own spec and a red-first test.

## 3. Class B — the correct mirrors, recorded so nobody reopens them

These are NOT work. **They live in two places, and this section said one until
review.** The first four are carried in `Ltx2UnportedPipelineFeature`
(`include/vllm/model_executor/models/ltx2_pipeline.h:1226-1261`) or in the
campaign's `Out` list. The last four are in NEITHER registry: they are refused at
their own call sites and recorded only there and in
`.agents/porting-inventory.md`. A reader sweeping the tree hits them and, finding
no mirror row, re-files them — the exact failure this section exists to prevent.

**The enum is not a class-B registry either, and the arithmetic says so.** It has
four enumerators, of which three are below; the fourth,
`kSpatiotemporalUpsampler`, is **class A** and is A8. Reading the enum as "the
mirrors" therefore both over- and under-counts.

| Item | Local anchor | Why it is a mirror, not a gap |
|---|---|---|
| `kBetaScheduler` | `ltx2_pipeline.h:1233-1238`, refusal `ltx2_pipeline.cpp:2226` | Upstream CONSTRUCTS IT NOWHERE. All seven `ltx-pipelines` entry points hard-code `LTX2Scheduler()`; vLLM-Omni has zero hits for the name. Mirroring that means no scheduler-kind field here either, so nothing reaches the refusal |
| `kInt8ConvRot` | `ltx2_pipeline.h:1239-1244`, refusal `ltx2_pipeline.cpp:2243` | Not an LTX-2 arm at all. `quantization_factory.py:22-26` is a `str` enum with `assert_never` at `:50` and four members, none int8. `convrot`/`quarot`/`spinquant` are 0 hits upstream |
| `kMultiGpuParallelism` | `ltx2_pipeline.h:1245-1260`, refusal `ltx2_pipeline.cpp:2266` | Four upstream forms, none of them CFG batching. Upstream's own `docs/multigpu/gemma.md:103-104` records that the DISTILLED pipeline this port runs takes no negative prompt and so "runs without CFG" |
| `kMultishot` (retired) | removed; provenance `ltx2_pipeline.h:1200-1221`, `ltx25-retire-dead-arms.md` §1.1 | FABRICATED. No such upstream entry point, symbol or string. Upstream's own shipped enhancer prompts instruct "Single continuous take -- no hard cuts". A defect in our record, not a gap in our port |
| `CFGStarRescalingGuider` | refusal `ltx2_pipeline.cpp:596-613`, header `ltx2_pipeline.h:299-307` | Appears in the LTX-2 tree exactly ONCE, at its own `class` statement (`guiders.py:31`). Control: `MultiModalGuider` occurs 110 times over `packages` and is CONSTRUCTED at 10 sites, `a2vid_two_stage.py:233`, `retake.py:297` and `ti2vid_two_stages_hq.py:274` among them |
| `LtxAPGGuider` | same refusal | Same: one occurrence, `guiders.py:78`, its own `class` statement |
| `LegacyStatefulAPGGuider` | same refusal | Same: one occurrence, `guiders.py:129`, its own `class` statement |
| `attn_res_x` decoder block | refusal `ltx2_video_vae.cpp:1281-1285`, scope note `:10-14` | Upstream CANNOT CONSTRUCT IT at this revision. `_make_decoder_block` passes `attention_head_dim=block_config["attention_head_dim"]` (`conv_video_decoder.py:94`) to `UNetMidBlock3D`, whose `__init__` (`resnet.py:210-222`) declares no such parameter and takes no `**kwargs` — the string occurs 0 times in the whole of `resnet.py` against 1 in `conv_video_decoder.py`. Upstream raises `TypeError`; this raises with the same reason named |

**Do not re-file these.** Each has been re-derived at least twice, and
`kMultishot` cost three review rounds to retire.

**The last four were re-derived for the first time here, and the count they
correct was published as "four correct mirrors" in §9.** They meet §2's class-B
definition verbatim — *we refuse because upstream refuses or never constructs it*
— and porting any of them would be inventing behaviour, which is why the local
refusals already say so. `## Gates` carries the upstream reads and both
controls.

## 4. Class A — the genuine capability gaps

Sized in **rows**, this project's unit: one issue's change plus the records it
invalidates. S = small (one focused change), M = medium (one recipe or one
leaf port), L = large (a new kernel or a new subsystem).

### A.1 The two that block the word "complete" outright

| # | Gap | Local anchor | Upstream anchor | Reached? | Size |
|---|---|---|---|---|---|
| **A1** | **The prompt-adherence gap. CORRECTNESS IS FAILING.** Our render scores 35.2719 CLIP against a 36.0087 bound -- FAIL by 0.7368 on S1 | `ltx25-prompt-adherence.md` `## Owed` | `ti2vid_one_stage.py` | yes, the default render | **M-L, mechanism unknown** |
| **A2** | **The speed gap: 3.230x the oracle.** Ours 302.954 s (n=3) against 93.8 s (n=1) | `ltx25-render-confirm.md` `## Outcome` | -- | yes, the default render | **L, two leaves = 54% of wall** |

**A1 is the gate, and AGENTS.md is unambiguous that it comes first:
"Correctness always comes first. Establish the declared token-exact gate before
you accept a performance result."** LTX-2.5 currently FAILS its declared
correctness gate. No amount of arm coverage makes the port complete while that
holds.

**A1 has no owner.** `ltx25-prompt-adherence.md` states it in its own `## Owed`:
"NOTHING OWNS CLOSING THE ADHERENCE GAP ... Owner of the REPAIR: **no row
exists**". The row that measured it does not fix it. Two waves are in flight
against the mechanism and neither has landed:

* `LTX25-ADHERENCE-BISECT` (PR #2520, issue #2514) localizes the gap to the
  LATENT and proposes a mechanism: our `one_stage` sigma shift anchors on 240
  tokens where upstream anchors on 4096 (`ti2vid_one_stage.py:207` passes no
  latent, so `schedulers.py:32` takes `MAX_SHIFT_ANCHOR`). Shifts 2.050000
  against 0.669271.
* `LTX25-ADHERENCE-DETAIL-LOSS` (PR #2525, issue #2513) **falsified** the
  smoothness hypothesis and **withdrew** a separable-upsampler lead that proved
  to be a DFT border artifact.
* `LTX25-SIGMA-SHIFT-MIRROR` is repairing the anchor and re-sampling six shipped
  arms' goldens.

So the mechanism is a live hypothesis, not a known quantity, and **n = 1 under
every adherence number**. The precise mechanism, which this section overstated
until review: the harness JUDGES render 1 alone (`ltx25-render-confirm.sh:471`,
`if [ "$i" = 1 ]`) and deletes only the PPM FRAMES of renders 2 and 3 (`:551`),
keeping each `phase-log-$i.json` (`:469`) — which is why the SPEED axis is n = 3
and the adherence axis is n = 1. A reading that moves by 0.74 between runs would
make the verdict a coin toss, and nothing excludes that.

**A2's shape is already measured**, by `LTX25-DEVICE-ARM-SURVEY` (landed). Two
leaves are 54.03% of the wall and both are host-pinned:

| arm | cost | shape |
|---|---:|---|
| Gemma-4 text tower, both passes | **82.336 s / 27.18%** | queue swap + one allocation + 49 downloads per pass |
| Connector, both passes | **81.338 s / 26.85%** | a ~386-line PORT |
| Video VAE non-convolution volume | 10.916 s / 3.60% | partial port |
| Audio VAE decode | 8.773 s / 2.90% | queue swap, coupled to a golden-changing accumulator change |

The DiT -- the only model actually on the accelerator -- is **4.99%**. The tower
arm was measured and deliberately NOT taken; it needs its own row and issue.

### A.2 The unported pipelines and arms

| # | Gap | Local anchor | Upstream anchor | Reached? | Size |
|---|---|---|---|---|---|
| **A3** | `DubItPipeline` | **no refusal anywhere**; one comment, `ltx2_pipeline.h:837` | `dubit.py` | **no** | **M** |
| **A4** | `HDRICLoraPipeline` | no refusal; cited `ltx2_lora.cpp:255` | `hdr_ic_lora.py:217` | no | **M-L** |
| **A7** | DiffVAE / `NADiffusionDecoder` | `ltx2_video_vae.cpp:1355-1359` and `ltx2_video_vae_tiled.cpp:393-397`, both `VT_CHECK` | `diffusion_video_decoder.py:62` | yes, refused by name | **L — needs a neighborhood-attention kernel** |
| **A8** | `kSpatiotemporalUpsampler` (both flags set) | refusal `ltx2_upsampler.cpp:465` | `model/upsampler/model.py:55-59` | **yes, the one REACHABLE unported-feature refusal** | **S** |
| **A9** | Upsampler `dims == 2` arm | refusal `ltx2_upsampler.cpp:467-472` (a `Require`) | `model/upsampler/model.py:85-100` | yes | **S** |
| **A10** | Multi-keyframe request surface | ABI has two scalar slots, `include/vllm.h:1080-1081` | repeatable `--image PATH FRAME_IDX STRENGTH`, `utils/args.py:805-817` | ABI ceiling | **M, ABI-additive** |
| **A11** | Image conditioning at CRF != 0 (H.264 round trip) | refusal `ltx2_image_preprocess.cpp:104-118` | `media_io/decode.py:430-434` | yes | **M — needs a vendored codec** |
| **A13** | Pre-2.3 flat vocoder arm | refusal `ltx2_loader.cpp:1757-1762` | `audio_vae/model_configurator.py:53-56` | yes | **S**, and arguably B |
| **A14** | Batched perturbation blend at batch > 1 | `ltx2.h:469`, `ltx2_device.cpp:320` | `attention.py:571-573` | degenerate at batch 1 | **S**, no consumer today |
| **A15** | `ICLoraPipeline` — reference-IMAGE and reference-VIDEO conditioning | refusal `ltx2_video.cpp:2708-2751` | `ic_lora.py:60` | yes | **M-L**, owed at `ltx25-ic-lora.md:356` ([#975](https://github.com/mudler/vllm.cpp/issues/975)) |
| **A16** | `conditioning_attention_strength < 1.0` and its attention mask | `Ltx2LatentState` carries no mask field; named at `ltx2_video.cpp:2740-2745` | `iclora_utils.py:151-160`, `mask_utils.py:170-243` | the default arm never builds a mask, so no | **M**, owed at `ltx25-ic-lora.md:357` ([#932](https://github.com/mudler/vllm.cpp/issues/932)) |
| **A17** | N-adapter LoRA fusion | refusal `ltx2_lora.cpp:252-257` | `fuse_loras.py:115`, `ic_lora.py` takes a list | yes | **M**, owed at `ltx25-ic-lora.md:358` ([#932](https://github.com/mudler/vllm.cpp/issues/932)) |
| **A18** | Reference-AUDIO conditioning delivery | refusal `ltx2_video.cpp:2754-2766` | `reference_audio_cond.py:34-65`, `utils/helpers.py:264-269` | yes | **M** — the op is ported and undriven |
| **A19** | Arbitrary-ratio audio resampling | refusal `ltx2_audio_vae.cpp:1051-1060` (a `VT_CHECK`) | `audio_vae/ops.py:36-42` | yes | **M** — gates every non-48 kHz input, so it gates A3 and `a2vid_two_stage` |
| **A20** | `LTXModelType.VideoOnly` / `AudioOnly` weight contracts | refusal `ltx2_device.cpp:1333-1335` (a `VT_CHECK`) | a different parameter set upstream builds | yes | **M** |
| **A21** | Text-to-audio on the ACCELERATOR | refusal `ltx2_video.cpp:5844-5847` | `t2a_one_stage.py:167` | **yes — a shipped, ABI-reachable arm that cannot use the GPU** | **M**, owed by [#1005](https://github.com/mudler/vllm.cpp/issues/1005) |
| **A22** | **Quantized COMPUTE.** Every encoding dequantizes to bf16 at load | §A.3; `ltx2_loader.cpp:469-546` | `quantization_factory.py:22-26` (four policies), `NVFP4Linear`, `fp8_scaled_mm`, `ltx-kernels/csrc/nvfp4/quantize.cu` | the load path is reached; no compute arm exists | **L** |
| **A23** | The prompt enhancer | declared out with NO citation, `ltx25-res2s-loop.md:271` | shipped upstream and reached from ten files, every pipeline among them | not built | **M**, and its exclusion is unratified |
| **A24** | **The phase-L6 dtype arms. Every non-DiT component runs f32 where upstream constructs it in bf16** | refusal `ltx2_text_encoder.cpp:54-61`; recorded owed in eight headers, §A.7 | `distilled.py:109` `self.dtype = torch.bfloat16`, passed at `:113,:122,:141,:148,:156,:165`; `:219-221` | **yes — reached at `:409` and `:546`, the second inside `Ltx2TextEncoderConditioning` on the render path** | **L, and it GATES §8 orders 4 and 5** |

**A3's anchor was wrong and its "reached" answer was wrong with it.** The cited
`ltx2_video.cpp:2753-2766` is the reference-AUDIO refusal, now A18. There is no
DubIt refusal anywhere in `src/` or `include/`: the only statement about it is
the comment at `ltx2_pipeline.h:837`, "`DubItPipeline`, which this tree does not
ship". A3 is therefore like A4 — a pipeline nothing refuses by name, so a user
asking for it gets the generic `(kind, version)` refusal at
`ltx2_pipeline.cpp:2197` and no word about what is missing.

**A5, A6 and A12 were in this table and are gone.** §A.4 records the two that
had already SHIPPED and §A.6 the one that moved to class E, because deleting a
wrong row without saying so is how the next survey re-files it. §A.5 records the
sweep that found A15 through A23.

**A8 is still the single cheapest real gap and the only reachable
unported-feature refusal in the enum.** It is the natural first bite.

### A.3 The quantized arms — stated explicitly, as the contract requires

**LTX-2.5 has checkpoint-format quantized arms and ZERO quantized compute.**

The loader reads four encodings -- BF16, F32, F8_E4M3 with an F32 scale, and U8
NVFP4 in two producer dialects -- and **every one of them dequantizes to bf16
during load** (`ltx2_loader.cpp:469-546`). The plan's own
`Ltx2DitQuant` field is written at `ltx2_loader.cpp:421,423,463` and **read by
no PRODUCT consumer**. The qualifier is load-bearing and this section said
"anywhere" until review: repo-wide the symbol occurs in six files, and three of
them are `tests/vllm/models/test_ltx2_loader.cpp` (29), `tests/vllm/multimodal/
test_ltx2_video.cpp` (5), `tests/vllm/models/test_ltx2_device.cpp` (2) and
`scripts/probe_ltx2_dit_load.cpp` (3). Under `src/` and `include/` it occurs
only in `ltx2_loader.cpp` (5) and its own header (3), against a positive control
(`Ltx2LoadDitFromSafetensors` occurs in six files across `src/` and `include/`).
Tests reading a field is not a product path branching on it, which is the claim
that matters -- and the loader says so itself at `:439`, "no consumer branches on
`quant`".

The consequence is worth stating plainly: **the quantized checkpoints buy
download and disk size, not VRAM and not speed.** Resident footprint is bf16 on
device (`ltx2_video.cpp:1068`) or f32 on CPU (`dit_options.widen_to_f32 =
!im.on_device`, `:1084`). The shipped first-party
NVFP4 DiT also carries 1176 `.input_scale` tensors -- it is a W4A4 file -- and
`IsScaleSidecar` (`ltx2_loader.cpp:360`, applied at `:396`) swallows every one of them, so the
activation half of that checkpoint's scheme is silently discarded.

The VAEs, upsampler and duration head are f32 only, and a quantized checkpoint
for them is refused by name (`ltx2_loader.cpp:1657-1660`). The text feature
extractor refuses any `compute_dtype` but f32 (`ltx2_text_encoder.cpp:54-61`),
as does the CPU reference DiT (`ltx2_dit.cpp:766`). The device DiT is the one
production path with a real narrow arm: `stream_dtype` defaults to `kBF16` and
accepts exactly `{kBF16, kF32}` (`ltx2_device.h:97-109`).

**The paragraph above is A24 stated as background, and background is not a
class.** Upstream's dtype is not a quantization policy: it constructs the text
tower, both VAEs, the latent upsampler and the duration head in **bfloat16**
(`distilled.py:109`), and this tree refuses every dtype but f32 on those paths
BY NAME, on a production-reached path. That is class A under §2's own test and it
is **A24**, not part of A22 — A22 is scoped to upstream's four quantization
policies and anchored on the DiT loader, and the DiT is the one component that
already HAS its bf16 arm (`ltx2_device.h:97-109`). §A.7 carries the derivation and
the reason it must be scheduled beside orders 4 and 5 rather than after them.

**Quantized compute is a class-A gap and this section gave it no number.** It is
**A22** in the table above and it is in the §8 sequence, because a capability
nobody numbered is a capability nobody schedules. Upstream ships four inference
policies (`quantization_factory.py:22-26`), an `NVFP4Linear` that reaches a
cuBLASLt GEMM, an `fp8_scaled_mm`, and its own quantize kernel
(`ltx-kernels/csrc/nvfp4/quantize.cu`). **In this row's favour: it does not
inflate A2.** `ltx2_oracle_manifest.json` records that the 93.8 s denominator
ran the bf16 arm, so the 3.230x is a bf16-to-bf16 ratio, and closing A22 would
move our side of it rather than correct it.

**GGUF k-quants: there is no path, no scaffold, no refusal string and no matrix
row.** Established across six independent spellings and search axes, with
positive controls -- `minimax_music3_quant.cpp` and `qwen3_dflash_gguf.cpp`
exist, no `ltx*gguf*` file does; `grep -i ltx .agents/quantization-matrix.md`
returns 0 against 28 `nvfp4|fp8` hits in the same file.

**This collides with a standing rule and the collision needs a decision.**
AGENTS.md: "GGUF k-quants are a standing requirement. They are not a choice for
each model." Five LTX specs have taken a "not applicable" exemption on the
ground that upstream ships no GGUF arm for any LTX-2 component
(`quantization_factory.py:23-26` with `assert_never` at `:50`) and llama.cpp does
not carry this architecture, so there is no behaviour to mirror and no
quant-matched comparison to serve.

**The "contradiction" this section reported is not one, and the correction
matters because it was the only evidence offered for the decision.** Read at
`781ea5487`: `ltx25-t2a-one-stage.md:568`, `ltx25-a2v-audio-input.md:422` and
`ltx25-a2vid-recipe.md:400` say "not applicable" flat.
`ltx25-generated-keyframes.md:387` says "not applicable to this key; owed for
LTX-2.5 as a whole under #644" and `ltx25-dfr-pipeline.md:370` says "not
applicable to this family ... Owed for LTX-2.5 as a whole under #644 only if
such a checkpoint appears". **Those two give the same two-level answer as each
other**, so the population is three flat readings and two that scope the
exemption to their own row while leaving the campaign question open. They do not
contradict; three are silent where two are explicit.

**What survives is the finding this row also made: no ratification record exists
for the campaign-level exemption.** Five specs assert it and no arbitrating
record was found in the tree. This row does not resolve it -- an exemption to a
standing requirement is a developer decision -- and returns it under
`## Owed`.

### A.4 What LEFT class A, and how the first pass got it wrong

Both entries below were removed because **the arm already shipped**, and in each
case the first pass reached its "no" by reading a line number carried over from
another record instead of opening the file. This is the row's own diagnosis
applied to the row.

**A5 — `TI2VidTwoStagesHQPipeline`. Shipped 15 days before this spec's base.**
It was cited at `ltx2_pipeline.cpp:1408` with *"Reached? no"*. That line is the
FIRST LINE OF THE HEADER COMMENT of `Res2sTwoStageRecipe`, which begins at
`:1454` and is dispatched at `:2112` as `res2s_two_stage` on version 2.5. It
landed in `4d7748646` (row `LTX25-RES2S-LOOP`, [#921](https://github.com/mudler/vllm.cpp/issues/921)),
an ancestor of this head. `docs/FEATURES.md:194` ships it with its gate — *"video
latents BIT-EXACT on 3 of 5 fixtures, 1 ulp on 2. 20 mutations, 18 DETECTED"* —
and `docs/models/ltx-2-5.md:114` passes `--pipeline-kind res2s_two_stage` inside
the very command block this spec's own D9 files a defect against. **The residual
is not a pipeline:** it is one out-of-scope ingredient, the per-stage distilled
LoRA strength, named at `ltx2_pipeline.cpp:1434`. It is owed below rather than
carried as a capability gap.

**A6 — DFR's temporal x2/x4 rounds loop. SERVED, 12 days before this base.** It was cited as a refusal at
`ltx2_video.cpp:2022-2027`. Those lines are audio guidance-extras assignments —
`audio->cfg_scale`, `audio->stg_scale` and their neighbours — and no refusal is
anywhere near them. The file says the opposite of the claim at `:2265`:
`// ── DFR's temporal x2/x4 rounds — SERVED (row LTX25-DFR-ROUNDS, #986) ──`. The
loop runs at `:5020` under `if (requested_temporal_rounds > 0)` at `:4982`, with
upstream's three refusals mirrored at `:2280-2317`. It landed in `d995c52f0`
([#986](https://github.com/mudler/vllm.cpp/issues/986)), also an ancestor of this
head.

**Both citations were plausible and neither was opened.** A line number in a
348-KB file is not evidence; the line is. Every anchor in this document has since
been re-resolved against `781ea5487`, and `## Gates` re-resolves the load-bearing
ones by content rather than by number.

### A.5 The sweep that closed the population, and the nine it found

§1's `Fail()` population was partial, and the first pass enumerated the refusals
it already knew about rather than sweeping for them. The sweep below is stated so
that a later reader can rerun it and get a population rather than a sample. It is
in `## Gates`.

Over the 52-file LTX source set, case-INSENSITIVELY (the first pass's regex was
case-sensitive and missed every `NOT ported`), for
`not ported|not implemented|not served|unported`: **98 lines**. Stripping the
scaffolding — the `allow_unported_modules` option, `UnportedFamilies`,
`RefuseUnported`, the enum and its dispatch, and the several "WHAT IS *NOT* THE
REASON" paragraphs that exist to record a claim that became false — leaves **21
capability sites**. Eleven were already classed. **Nine were in no class at all**,
and are now A15 through A23.

**The 98 is a command and the 21 is a READING, and only the first re-runs.**
`## Gates` prints the 98 and the filter that removes the named scaffolding; what
turns the remainder into "capability sites" is a judgement over each line. A later
reader re-reads it rather than re-running it, and the count moves if the judgement
does.

**This section then said "the tenth unclassed site is a records defect ... and is
D11", and that closure claim is FALSE.** Reading all 98 lines, and running a
second pattern set the first sweep never carried — `\bowed\b|does not ship|this
tree does not|out of this phase`, **69 lines** over the same 52 files, of which
the union with the first is **165** — gives **nine** statements that describe as
unported or owed something this tree ships. D11 is one of them. It is now a
CLUSTER of nine in §6, with the tree's falsifier beside each, because "one" was
the same shape of error as the partial sweep this section exists to correct: a
population stated as closed by a pass that had not swept it. Both pattern sets
bound it, and a stale statement neither matches is still outside it.

Three of the nine deserve their reason stated, because they are the ones a survey
is most likely to miss again:

* **A15/A16/A17 are one owed table this row walked past.** `ltx25-ic-lora.md`
  lists all three under its own `## Owed` at `:356-358`, with their issues. This
  spec touched the first only as an unnumbered sub-cost of A3, and never named
  the other two. An owed table in another spec is exactly where a completion
  survey has to look, and looking there was this survey's whole method.
* **A19 gates two entries above it.** Refusing every sampling rate but the audio
  VAE's own means any real audio input has to arrive pre-resampled, which is what
  A3 (DubIt) and the shipped `a2vid_two_stage` arm both need. A gap that gates
  other gaps belongs in the sequence before them.
* **A21 is a shipped arm that cannot use the accelerator.** `t2a_one_stage` is
  ABI-reachable and documented, and `Ltx2DitForwardDevice` dereferences `*video`
  unconditionally, so the device path refuses it by name. It bears directly on
  A2: a wall-clock survey that never ran this arm on the GPU has not measured it.

### A.6 Class E — the one gap that fails class A's own test

**A12, the prompt K/V cache on the device forward, was misfiled and is now E1.**

Class A's test is "does upstream construct this?" Upstream does not. Read at
`fd4ded7f`, `transformer.py:437-440` is a COMMENT observing that with
`use_prompt_adaln_single=False` the K/V modulation is "timestep-independent and
cacheable across denoising/AR steps"; `:441-443` then compute `kv_modulation`
unconditionally. No cache is built. `kv_cache`, `prompt_kv`, `cached_kv`,
`text_kv` and `PromptCache` return **zero files** across `ltx-core` and
`ltx-pipelines`, against a positive control of 66 files matching `attention` in
`ltx-core` alone. The only `kv_cache` upstream is an unrelated VAE
neighborhood-attention slab cache in `ltx-kernels/vae/block_fna_dsl.py`.

| # | Gap | Local anchor | Upstream | Reached? | Size |
|---|---|---|---|---|---|
| **E1** | Prompt K/V cache on the DEVICE forward, which the HOST forward has | refusal `ltx2_device.cpp:1336-1340` (a `VT_CHECK`); host path `Ltx2DitForward` | **none — upstream builds no such cache** | yes | **M** |

**It is still a real gap, and that is why it is recorded rather than dropped.**
`Ltx2DitForward` caches on the host and `Ltx2DitForwardDevice` refuses a cache by
name, so the same request costs more on the accelerator than off it. A user can
feel that. What it is NOT is an upstream-parity gap, and calling it one would have
put a local optimisation into a list whose whole claim is "upstream implements
this and we do not".

### A.7 A24 — the phase-L6 dtype arms, and why no refusal sweep could find them

**Upstream resolves ONE model dtype and it is bfloat16.** `distilled.py:109`
sets `self.dtype = torch.bfloat16` and hands that same object to every component
the pipeline constructs: `PromptEncoder` (`:113`), `ImageConditioner` (`:122`),
`DiffusionStage` (`:129`), `VideoUpsampler` (`:141`), `VideoDecoder` (`:148`),
`AudioDecoder` (`:156`) and `DurationPredictor.from_checkpoint` (`:165`). The
call path repeats it: `dtype = torch.bfloat16` at `:219`, and `:220-221` gives
`vae_dtype` the same value when the caller passes none.

**This tree runs f32 on every one of those paths except the DiT.** The text
tower's refusal is by name and on a production-reached path:

```text
src/vllm/model_executor/models/ltx2_text_encoder.cpp:54  void RequireF32(vt::DType dtype)
  "compute_dtype must be f32. The bf16 / FP8 / NVFP4 arms of the text tower
   are phase L6 of .agents/specs/ltx-2-5.md and are NOT implemented..."
```

It is reached at `:409` and `:546`; `:546` is the first statement of
`Ltx2TextEncoderConditioning` (`:541`), which `Ltx2EncodePromptToConditioning`
calls at `:1196`, which `ltx2_video.cpp` calls at `:2501`, `:3241` and `:5798`.
That is a production entry point, not a test.

**The derivation of the component count, from ten matched files down to eight
components.** Joining lines — the marker is split across a line break in one header and a
line-based grep silently drops it — **10 of the 52 LTX files carry the phase-L6
marker**. One, `ltx2_loader.h:5`, carries it in a `Row:` provenance line and
records no dtype debt. That leaves **nine components whose own header records
the narrow arm as owed**, and one of those nine is the DiT (`ltx2.h:33-39`),
whose PRODUCTION path is `Ltx2DitForwardDevice` and already ships bf16
(`stream_dtype` defaults to `kBF16`, `ltx2_device.h:97-109`); `ltx2.h` is the CPU
REFERENCE arm and its remaining narrow arms are A22's. **A24 is therefore the
other eight:**

| component | header record | what upstream constructs it as |
|---|---|---|
| Text tower / feature extractor | `ltx2_text_encoder.h:82-92` | `PromptEncoder`, `distilled.py:113` |
| Embeddings connector | `ltx2_connector.h:47-51` | inside `PromptEncoder` |
| Video VAE decoder | `ltx2_video_vae.h:47-54` | `VideoDecoder`, `:148` |
| Video VAE encoder | `ltx2_video_vae_encoder.h:52-57` | `ImageConditioner`, `:122` |
| Video VAE device kernels | `ltx2_video_vae_kernels.h:44-51` | the same decoder's storage |
| Tiled-decode buffer | `ltx2_tiling.h:88-94` | `latent.dtype`, `conv_video_decoder.py:427-431` |
| Latent upsampler | `ltx2_upsampler.h:66-70` | `VideoUpsampler`, `:141` |
| Duration head | `ltx2_duration_head.h:55-58` | `DurationPredictor`, `:165` |

**One component upstream builds in bf16 is deliberately NOT in that table, and
naming it is the point of stating the rule.** Upstream constructs `AudioDecoder`
in `self.dtype` too (`distilled.py:156`), and this tree runs the audio VAE in f32.
It carries no phase-L6 marker because `ltx2_audio_vae.cpp:7-12` ARGUES its f32
rather than owing it: upstream's own BWE path forces the whole vocoder chain to
float32 whatever the weight dtype, because bf16 accumulation through ~108
sequential convolutions degrades its spectral metrics by 40-90%
(`vocoder.py:575-580`), and this port extends that one contract up to the
spectrogram decoder above it. **Whether that extension is sound is a question the
marker sweep cannot answer** — upstream pins f32 at the vocoder and not at the
decoder — so it is named here as a bound on A24's population rather than folded
into the eight or dropped. It is returned under `## Owed`.

**No refusal sweep could have found this, and that is the point.** §1's four
populations count `Fail(`, `Refuse(`, `VT_CHECK(` and `Require(`; §A.5 sweeps four
"not ported" phrasings over them. A24 has exactly ONE refusal string in the whole
set (`ltx2_text_encoder.cpp:58`, and §A.5's sweep did match that line) — the other
seven components record the debt in a HEADER COMMENT and then simply compute in
f32. A capability gap that refuses nothing is invisible to a survey that
enumerates refusals, which is the same shape of error §A.5 corrected one level
up. `## Gates` carries the marker sweep with its denominator and its exclusion.

**It bears directly on A2, and that is why it is sized L and placed where it
is.** §A.1's two host-pinned leaves are **54.03% of the wall**, and both of them —
the Gemma-4 text tower and the connector — are in the table above. §8 orders 4
and 5 land those two ports. Landing them in f32 moves twice the bytes upstream
moves, on the two paths that dominate the wall, and AGENTS.md names exactly this
failure under *Inherit vLLM defaults*: **"A token gate cannot detect a dtype that
is too wide."** The goldens would still pass. So A24 is not work that follows the
speed work; it is the dtype those two ports are landed in.

**What it is NOT.** It is not A22. A22 is upstream's four *quantization policies*
(`quantization_factory.py:22-26`), it is anchored on the DiT loader, and its
arms are FP8 and NVFP4. bf16 is upstream's DEFAULT dtype and reaches every
component, DiT included. Merging the two would hide a default inside a
quantization row.

## 5. Class C is empty — the three blockers, verified rather than inherited

Each was read against `git show fd4ded7f:<path>` and against the local tree at
`63889449c`.

### 5.1 `TI2VidTwoStages` — ALREADY LANDED

The blocker said two checkpoints were absent from the NAS. **The arm shipped at
`affc2a7fd`** ("the plain two-stage pipeline, on the schedule anchor upstream
actually uses (#1093) (#1156)"). `Ti2VidTwoStageRecipe` is at
`ltx2_pipeline.cpp:1753` and is dispatched for versions 2, 2.3, 2.4 and 2.5 at
`:2141-2160`. Both named checkpoints are on the NAS and recorded with byte counts
and revisions: `docs/USAGE.md:622` (dev transformer, 42,018,190,584 B) and `:625`
(distilled LoRA, 8,899,889,568 B). What remains is one GPU lease for a
real-weights render against upstream -- scheduling, not a blocker.

### 5.2 `DubIt` — the clamp is in a different function, and it is upstream's own

The blocker said our one ported shift "structurally cannot" produce a negative
position because it clamps at zero. **The clamp is real and it is in the wrong
function for the claim.** `std::max(0.0, ...)` sits at
`ltx2_conditioning.cpp:600`, inside `Ltx2ConditionVideoByReference` (opens
`:564`) -- the VIDEO reference item. It is a faithful port of upstream's own
`torch.clamp(..., min=0)` at `reference_video_cond.py:69-74`, whose comment names
the intent. **Removing it would be a parity defect.**

The path DubIt uses is `Ltx2ConditionAudioByReference` (opens `:615`), which
takes caller positions and passes them through with no sign check, no clamp and
no unsigned type. RoPE downstream is signed arithmetic over `const double*`
(`ltx2.cpp:609-645`). There is no structural obstacle.

The in-tree refusal never mentions a clamp. `ltx2_video.cpp:2753-2766` says what
is actually missing: "the CONDITIONING ITEM's delivery ... nothing in this phase
loop constructs one from a request". Size: a ~5-line shift helper, request
plumbing, a recipe row, and the reference-video pixel path. **One M row.**

### 5.3 `HDRICLora` — the "deliberate exclusion" citation is a misread table cell

The blocker cited `ltx25-retire-dead-arms.md:167` as a scope decision excluding
colour science. **That line is a cell in a three-column table whose header is
"Is it a generation mode?"**, answering that question about the `scene-linear`
hit with "no -- colour science". It disposes of `kMultishot`, not of HDR. The
campaign's own `Out` list (`ltx-2-5.md:290-292`) names DiffVAE, the temporal
upsampler, LoRA fusion, multishot, `int8-convrot` and multi-GPU, and **the HDR
pipeline is not among them**. It IS named two lines later: `ltx-2-5.md:294-296`
lists `HDRICLoraPipeline` among four upstream pipelines that are out, "recorded
as owed, not silently dropped", with an issue. So the precise claim is the
narrow one — **no scope decision excludes HDR COLOUR SCIENCE**, and the cited
cell is not one — and this section overstated it as "HDR is not in it" until
review. The misreading itself has propagated to `ltx25-retake.md:212` and
`:499`.

The colour science actually needed is bounded: an inverse LogC3 (`hdr.py:60-65`,
six lines), optionally an inverse ACEScct (`:75-79`, five lines), and a primaries
rotation (`:140-152`) that is **identity on the default LogC3 path**, Rec.709 to
Rec.709. Upstream does no tone mapping in the pipeline at all --
`hdr_ic_lora.py:3-5` says the pipeline returns linear HDR and "tonemapping and
EXR saving are the caller's responsibility". The conditioning input path needs
none of it: `compress_ldr` is an identity clamp (`hdr.py:122-124`). **~15 lines
of scalar float math**, plus the pipeline itself and an HDR-capable frame writer.
**One M-L row.**

### 5.4 What this means

**Nothing in LTX-2.5 is externally blocked on the engine side.** The genuine
external dependencies that remain are of a different kind and belong in class D:
GPU lease availability, three unpublished or unrecorded checkpoints, and an
oracle denominator that is n = 1.

## 6. Class D — measurement and records debt

No engine change. **Twelve clusters**, D1 through D12; the count and the list
disagreed until review, which is the same defect the clusters are about. **D11 is
a cluster of nine and was published as one item**; §6.1 carries all nine.

| # | Debt | Anchor | Size |
|---|---|---|---|
| **D1** | **n = 1 under every adherence and speed number.** The oracle denominator has no spread of its own; ours is measured at 8.03% | `ltx25-render-confirm.md`, `ltx25-prompt-adherence.md` | lease time |
| **D2** | `ltx25-render-confirm.sh` never passes `--adherence-model`, so the next lease re-takes the PPM/blockiness verdict (`:477-480`) and not the adherence one. Evidence is an absence with a control: 0 hits for `adherence` in the harness against 4 for `--reference`, while `ltx25-render-compare.py:79` documents the flag | `scripts/ltx25-render-confirm.sh`, `scripts/ltx25-render-compare.py:79` | S |
| **D3** | **The quantization matrix carries ZERO LTX rows** (control: 28 `nvfp4|fp8` hits for other models in the same file) | `.agents/quantization-matrix.md` | S |
| **D4** | **Four checkpoints have no `docs/USAGE.md` row**: the spatial x2 upsampler, the temporal x2 upsampler, the IC-LoRA, and DFR's `keyframe_slot_sft` base. `grep -ci 'upsampl\|upscal' docs/USAGE.md` = **0** against an `ltx` control of 16. The spatial upsampler is REQUIRED by the DEFAULT `distilled_two_stage` recipe, so a reader cannot feed the default arm | `docs/USAGE.md:608-630` | S |
| **D5** | Two rows carry no digest where the file's own local standard asks for one on every LTX row | `docs/USAGE.md:623,625` vs `:598-607` | S |
| **D6** | `docs/USAGE.md` names no pipeline-arm refusals -- not `dmd2`'s unreachability on 2.5, not the per-kind version refusals, not the unported-feature refusals | `docs/USAGE.md` | S |
| **D7** | **The campaign `Out` list is stale in five of its entries.** The temporal upsampler, LoRA fusion, multishot, `KeyframeInterpolation` and `TI2VidTwoStages` have all landed or been retired | `ltx-2-5.md:288-296` | S |
| **D8** | **Three stale blocker bullets** assert what §5 falsifies, plus two propagated copies. They also carry the two wrong anchors this spec inherited (`hdr_ic_lora.py:229` at `:986`, `ltx2_lora.cpp:246` at `:987`; true values 217 and 255) and a third, the reference-audio refusal cited at `ltx2_video.cpp:1991-2004` when it is at `:2754-2766` | `ltx-2-5.md:973-996` (**not** `:962-991`, which is an unrelated #1458 paragraph), `ltx25-retake.md:212,499` | S |
| **D9** | The copy-pasteable render command omits `--checkpoint-class` and refuses as written | `docs/models/ltx-2-5.md:105-118` | S |
| **D10** | **The owed-backlog gate misses 7 of the 166 issue numbers under an owed heading** — a numbered or differently-worded heading is skipped entirely (`:2012`), and only the first section is read (`:2014`). Measured in the gate's OWN unit: `owed_issues()` returns `set[str]` of issue numbers and never counted items, so this row's earlier "~74 of 299 items" measured a quantity the function does not compute. §2.2 | the record checker's `owed_issues()` reader (retired, `a9f6186c2`) | S, needs a red-first test |
| **D11** | **NINE in-tree statements describe as unported or owed something this tree ships.** Published as one until review; the table below carries all nine with the tree's falsifier beside each. §A.5 | nine sites, listed below | S each, but see the note |
| **D12** | **The campaign carries no ratification of the GGUF k-quant exemption**, which five specs assert independently. §A.3. This is records debt only in that no record arbitrates; the decision itself is the developer's | five `ltx25-*.md` specs; no arbitrating record found | decision |

### 6.1 D11 in full — the nine statements the tree falsifies

**This was published as a single defect and it is a cluster.** The sweep that
found it is in §A.5 and its commands are in `## Gates`. Every falsifier below is a
line in this tree, and four of the nine are falsified by ONE landing:
`d995c52f0` (row `LTX25-DFR-ROUNDS`, [#986](https://github.com/mudler/vllm.cpp/issues/986)),
the same commit §A.4 cites to delete A6.

| # | The statement | Where | What falsifies it |
|---|---|---|---|
| **D11a** | the cross-attention perturbations "exist upstream (SKIP_A2V_CROSS_ATTN, SKIP_V2A_CROSS_ATTN) and are NOT ported" | `ltx2.cpp:885` | `Ltx2PerturbationType::kSkipA2vCrossAttn`, `ltx2_pipeline.h:397`; `docs/FEATURES.md:197` ships them. The refusal is still CORRECT — a perturbed pass carrying a cross flag should refuse — only its stated reason is false |
| **D11b** | of the temporal x2 upsampler: "no phase of any recipe this engine serves consumes it, because its only upstream consumer is DFRPipeline's rounds loop, which is not ported" | `ltx2_video.cpp:3479-3486`, a user-facing `Fail()` | The rounds loop is PORTED and runs at `:4982-5040`, calling `Ltx2UpsampleVideoLatent(im.temporal_upsampler_cfg, ...)` at `:5040`. **The worst of the nine:** D11a's refusal is right for a wrong reason, but this one tells a DFR user holding the temporal checkpoint that a shipped arm does not exist |
| **D11c** | "the engine's ONE upsampler call site is the `kSpatialUpsample` phase input transform ... and upstream's only consumer is `DFRPipeline`'s rounds loop ..., which is not ported" | `ltx2_upsampler.h:27-31` | Three call sites: `ltx2_video.cpp:3505`, `:3532` and `:5040`. Same false clause as D11b, in a second file |
| **D11d** | `temporal_upsample_rounds` is "DEFINED ... still not SERVED", "defined so that its own refusal can name the missing loop" | `ltx2_video.cpp:2153-2155` | The knob is parsed at `:2317` into `requested_temporal_rounds` and drives the loop at `:4982`. The SAME comment records the sibling knob as "SERVED by row LTX25-DFR-PIPELINE #986" two lines above |
| **D11e** | of the ported temporal-only x2 upsampler: "Nothing shipped drives the ported arm yet, so it is gated, not served" | `ltx2_pipeline.h:1164` | `:5040` drives it. This one understates reach, which is the direction AGENTS.md's "Nothing lands dead" gate cannot catch |
| **D11f** | "the one construct that passes true is `VideoGeneratedKeyframeSlots` (keyframe_slots.py:121), which is not ported" | `ltx2_conditioning.cpp:64-67` | `Ltx2ConditionVideoByGeneratedKeyframeSlots` is defined in the SAME file at `:154` and called at `ltx2_video.cpp:4102` |
| **D11g** | "The ANALYSIS half (`AudioEncoder` ... and the mel front-end `AudioProcessor`) ... is NOT ported here — it is owed, and the same is true of the video VAE's encoder" | `ltx2_audio_vae.cpp:14-19` | `Ltx2AudioEncoderForward` is defined in the SAME translation unit at `:1142`, under that file's own banner at `:887-889` ("THE ENCODER HALF ... which phase L4 recorded as owed"), and is reached at `ltx2_audio_input.cpp:203`. `ltx2_video.cpp:3032` says so in a refusal string |
| **D11h** | "The ENCODER half is out of this phase and owed" | `ltx2_video_vae.cpp:17` | `Ltx2ConvVideoEncode` is defined in the SAME translation unit at `:1565`, under that file's own banner at `:1364-1365`, and is reached at `ltx2_video.cpp:3108` and `:3756`. **Outside §A.5's four patterns** — only the second pattern set reaches it |
| **D11i** | `_guided_denoise` "is the piece four unported pipelines are each blocked on — `a2vid_two_stage.py:230`, `ti2vid_two_stages.py:248`, `ti2vid_two_stages_hq.py:271`, `keyframe_interpolation.py:232`" | `ltx2_denoisers.h:12-15` | All four dispatch: `a2vid_two_stage`, `ti2vid_two_stage`, `res2s_two_stage` (§A.4's port of `ti2vid_two_stages_hq`) and `keyframe_interpolation` are four of the ten kinds §7 counts. The WEAKEST of the nine — the sentence is a historical framing in the present tense — and it is listed because a reader cannot tell that from the line |

**D11g and D11h are the two that should not be possible.** Each contradicts a
banner in its own file, a few hundred lines below the header that carries it. A
file whose top says the encoder is owed and whose middle says "THE ENCODER HALF
... which phase L4 recorded as owed" and then defines it is not a record that
drifted from the tree; it is a record that drifted from itself.

**Sizing.** Each of the nine is a comment or message repair, S. What is NOT S is
the sweep: closing D11 means reading 165 lines rather than editing nine, and the
population is bounded by two pattern sets rather than closed.

## 7. Reachability — what a production path can actually reach

LTX-2.5 **is** on the C ABI, but only as a string value. `include/vllm.h`
contains **zero LTX-named symbols** (7 case-insensitive `ltx` hits, all in
comments; spellings tried: `ltx`, `LTX2`, `ltx_2`, `ltx-2`, `ltx25`, `ltx2p5`,
`ltxv`, `lightricks`). It rides the generic video seam:
`REGISTER_VLLM_VIDEO_FAMILY(ltx2, kLtx2VideoFamily, DetectLtx2Video, LoadLtx2VideoFamily)`
at `ltx2_video.cpp:5926`, family string `"ltx-2.5"`
(`include/vllm/multimodal/ltx2_video.h:131`).

**Ten pipeline KINDS and 28 accepted `(kind, version)` PAIRS** are resolved by
`ResolveLtx2PipelineRecipe` (declared `ltx2_pipeline.cpp:2082`, dispatch
`:2087-2196`); everything else falls through to one refusal at `:2197-2201`.
This section said "ten `(kind, version)` recipe rows" until review, conflating
the two counts: `one_stage`, `a2vid_two_stage`, `ti2vid_two_stage`,
`keyframe_interpolation` and `t2a_one_stage` each accept four versions, while
`dfr` and `res2s_two_stage` accept 2.5 alone. `## Gates` counts both. The server registers `/v1/videos` only
when `--video-dit` is non-empty, so **LTX-2.5 is not reachable on the server's
own default configuration** -- and with video enabled the default arm is
`distilled_two_stage`/`2.5` alone.

**A third arm is ABI-reachable and ACCELERATOR-unreachable:** `t2a_one_stage`
refuses the device forward by name (A21, `ltx2_video.cpp:5844-5847`), so the one
audio-only guided arm runs host-side whatever the caller asks for.

**Two arms are ABI-reachable and server-UNREACHABLE**, because
`VideoGenParamsFromRequest` (`video_engine.cpp:349-383`) forwards no
per-generation extras at all: `a2vid_two_stage` needs `audio_path`, and `retake`
needs its three retake extras. Any server render carrying an image conditioning
also refuses, because `image_crf` defaults to the checkpoint's 18. That is
**A10-adjacent and belongs in the sequencing**: it is the difference between "we
shipped the arm" and "a user can ask for it".

`dmd2` is accepted by the table only at versions 2 and 2.3, and every recorded
LTX checkpoint declares 2.5, so **`dmd2` is unreachable with any checkpoint this
project records**.

## 8. The sequence

Correctness first, because AGENTS.md requires it and because a speed number
taken on a failing tree measures the wrong thing.

| Order | Work | Depends on | Size |
|---|---|---|---|
| **1** | **A1 — close the adherence gap.** Land the sigma-shift repair, then re-score | #2520, #2525, `LTX25-SIGMA-SHIFT-MIRROR` in flight | M-L, mechanism unconfirmed |
| **2** | **D1/D2 — get n > 1** on both axes and wire the harness to score adherence in-lease | lease | S + lease time |
| **3** | **A24 — resolve the dtype the two leaves are ported IN.** The text tower and the connector are two of A24's eight components; landing 4 and 5 in f32 lands them at twice upstream's bytes, and no token gate can see it | 1, 2 | **L**, and orders 4-5 carry the first two components of it |
| **4** | **A2a — the text tower queue swap**, 27.18% of the wall, the cheapest shape in the ranking | 1, 2, 3 | M |
| **5** | **A2b — the connector port**, 26.85%, a ~386-line port | 4 (so the two are measured apart) | L |
| **6** | **A8, A9** — the two upsampler refusals, the cheapest real gaps | none | S each |
| **7** | **A19 — arbitrary-ratio audio resampling.** Before A3 and A18, because it gates both | none | M |
| **8** | **A15, A16, A17 — the IC-LoRA family**, one owed table with its issues already filed | A16 needs a mask field on `Ltx2LatentState` | M-L total |
| **9** | **A3 — DubIt**, plus a refusal that names it instead of the generic one | 1, 7, 8 (it needs an IC-LoRA) | M |
| **10** | **A18 — reference-audio delivery.** The op is ported and undriven | 7 | M |
| **11** | **A21 — text-to-audio on the accelerator.** A shipped arm that cannot use the GPU, and A2 cannot measure it until it can | 5 | M |
| **12** | **A10 + the server extras ceiling** — make the shipped arms askable | ABI-additive | M |
| **13** | **E1, A14, A13, A20** — device prompt-KV cache, batched blend, flat vocoder, the one-modality weight contracts | 5 | S-M each |
| **14** | **A4** — HDRICLora, pipeline plus an HDR frame writer | 1 | M-L |
| **15** | **A11** — CRF round trip, needs a vendored codec | codec decision | M |
| **16** | **A22 — quantized compute.** The one item that changes the resident footprint and the only one that could move A2 by more than the two host leaves | 3, 5, and a ratified quant plan | **L** |
| **17** | **A7 — DiffVAE.** A new neighborhood-attention kernel. The largest single item and the last, because nothing depends on it | none | **L** |
| **18** | **D3-D12** — the records sweep, ridden along on the changes that invalidate each. D11 is nine statements, not one (§6.1), and D11b is user-facing | each item's own change | S each |

**A24 sits at 3 and not after 5, and the placement is the finding.** It is not
"do the dtype work first"; it is that orders 4 and 5 port the two components that
are 54.03% of the wall, and a port lands in some dtype whether or not anybody
chose one. AGENTS.md under *Inherit vLLM defaults*: **"A token gate cannot detect
a dtype that is too wide."** The goldens pass either way, so the choice has to be
made before the port rather than discovered after it. The remaining six
components follow with their own leaves.

**A23 (the prompt enhancer) is deliberately absent from this sequence.** It is
in class A because upstream ships it and we do not, but its exclusion was
declared without a citation and has never been ratified. Scheduling it would
answer a question that belongs to the developer; it is returned under `## Owed`
instead. The same reasoning applies to D12.

## 9. The number

**Twenty-one class-A rows, of which two are the gate, four are L, and ten were
missed by the first pass; one class-E row; eight correct mirrors; zero external
blockers; twelve records clusters, one of which is itself nine statements; a
352-item internal owed backlog under a stated rule; and a correctness verdict
that is currently FAILING.**

**The row count, derived.** Class A sizes read off §4: 4 rows at S (A8, A9, A13,
A14), 10 at M (A3, A10, A11, A16, A17, A18, A19, A20, A21, A23), 3 at M-L (A1,
A4, A15) and 4 at L (A2, A7, A22, A24), plus E1 at M. Taking S and M at one row
each, M-L at one to two, and L at two to four, and adding one to two for the
records work that does not ride another change (D10 needs its own spec and a
red-first test; D12 is a decision, not a row):

> **27 to 39 rows.** It is a floor, not a range with a ceiling.

**The range moved because A24 entered at L, and it moved by more than one row.**
That is the arithmetic working as intended: an L is two to four rows, so one
addition at that size widens the range at both ends. The previous 25 to 35 was
taken over twenty class-A rows and is superseded, not corrected.

**No duration follows from it, and this row withdraws the one it published.**
§9 previously said "six to ten weeks at the campaign's observed cadence" beside
its only cadence figure, "roughly 30 landed rows in 15 days". Those two
statements contradict each other by about a factor of four: 30 rows in 15 days
applied to 22-30 rows is 11 to 15 days, not 42 to 70. The arithmetic was never
shown, so nobody could see the contradiction, and the range had already been
quoted onward.

The cadence itself is measurable and is measured here, with its command in
`## Gates`: **43 distinct LTX row IDs appear in the conventional-commit scopes
of the 155 LTX-matching non-merge commits reachable from this head since
2026-08-17**, over 16 calendar days. That is the raw fact. **Multiplying it by
the row count would be wrong, and the reasons are structural rather than
cautious:**

1. **The two populations differ in kind.** The observed 43 are weighted toward
   S rows, spec edits and record repairs. Of the 21 rows remaining, **four are
   S** and four are L. A rate measured on one distribution does not transfer to
   the other.
2. **A1 has no duration because it has no mechanism.** #2525 falsified the
   leading hypothesis and withdrew a second; #2520 has a live one. An open
   research question is not a row with a length.
3. **Several rows are lease-bound, and a lease is a queue rather than a rate.**
   D1, D2 and every re-score wait on GPU availability, which no commit cadence
   measures.
4. **"Row" is not a constant unit.** Some of the observed 43 are one-line record
   repairs and some are whole ports.

**So the schedule is returned as a decision, not answered.** `## Owed` carries
it. A number this row cannot derive from a stated, runnable rule does not belong
in a document whose entire subject is numbers that were propagated rather than
derived.

**Three things make the row count a floor and not an estimate.**

1. **A1 has no mechanism.** #2525 falsified the leading hypothesis and withdrew
   a second. #2520 has a live one. If the sigma anchor is not the cause, A1 is
   an open research question and no schedule survives it.
2. **A2 has no ceiling, and AGENTS.md forbids declaring one.** Moving the two
   host-pinned leaves recovers at most 54% of the wall; 3.230x does not become
   1.0x by arithmetic. What is left after them is unmeasured, and A22 —
   quantized compute — is the only enumerated item that could move it further.
3. **n = 1 under both.** Neither the adherence FAIL nor the 3.230x has a
   demonstrated spread on the oracle side. D1 could move either verdict.

**And a fourth, which this review established:** class A grew by six the first
time anybody ran a closed sweep over the refusal surface, and by a seventh — A24
— the first time anybody swept something the refusal surface does not contain.
§"Risks" called class A a lower bound as a caveat; it is now a measurement, and
A24 says what the bound is a function of: the PATTERNS, not the tree. Seven of
A24's eight components refuse nothing at all.

**An honest summary in one line:** LTX-2.5 is a broad, deeply documented port
whose arms are nearly all present and whose *correctness gate is failing with no
owner and no mechanism*. The remaining arm work is ordinary and sequenceable.
The gate work is not, and it is what "complete" waits on.

## 10. What this row could not determine

Named rather than filled in.

* **Whether the GGUF exemption was ever ratified.** Five specs assert it
  independently — three flatly, two scoped to their own row — and no arbitrating
  record was found. The "contradiction" this row first reported between them was
  not one; §A.3 records the correction. What is genuinely absent is the
  ratification.
* **Issue state for most of the campaign.** `gh` is scattered-blind at this
  account: #435, #644, #1093, #1094, #1095, #1096, #1854 and #2295 all return
  "Could not resolve" while #2513, #2514, #2520 and #2521 resolve normally, and
  a full `gh issue list` returns 140 items. **`REMOTE_UNVERIFIED`, and never
  absence** -- the same defect that once got 817 issues wrongly recorded as
  deleted. Every issue number in this spec is cited from the tree, not from the
  forge.
* **Whether A1 and A2 interact.** The sigma-shift repair re-samples six arms'
  goldens; nobody has measured what it does to the wall.
* **The two disagreeing bf16 footprint figures** in one header
  (`ltx2_loader.h:463` "~21 GB bf16 for the shipped FP8 DiT" against `:504`
  "~39 GB bf16"). A real inconsistency; not reconciled here.
* **How long the remaining 27 to 39 rows take.** §9 derives the count and
  refuses the duration. Nothing in this tree measures a rate that transfers to
  the remaining population.

## Risks

* **This spec is a projection and will rot.** Every count is SHA-stamped and
  `## Gates` re-derives them. A reader at a later SHA reruns them first.
* **Sizes are shapes, not estimates.** They are read off the code and the
  measured phase table. None is a schedule commitment.
* **Class A could grow, and on review it did — from fourteen to twenty-one.** It
  is a lower bound, not a caveat: nine additions came from the first CLOSED sweep
  over the refusal surface, and the tenth (A24) from a sweep over a marker that
  surface does not carry at all. The population every sweep covers is defined by
  its grep patterns over 52 files. An arm upstream implements that no pattern in
  `## Gates` matches is still not in it, and A24 is the proof that such arms
  exist: seven of its eight components refuse nothing, so no refusal count could
  ever have reached them. **Class B is a lower bound for the same reason** — it
  grew from four to eight when the guider and VAE-block refusals were read.
* **Every count here is a property of the rule that produced it.** §2.1's item
  count reads 299, 350, 352, 362 or 368 under five different rules over the same
  tree. Cite the rule with the number, or cite neither.

## Gates

Re-derive at any SHA; each prints its own denominator. `$LTX` is the 52-file LTX
source set every count in §1, §A.5 and §7 is taken over.

```sh
# The file set itself, named once so no count is taken over a different one
LTX=$(git ls-files | grep -E '(^src/vllm/(model_executor/models|multimodal)/ltx2.*\.(cpp|h)$)|(^include/vllm/(model_executor/models|multimodal)/ltx2.*\.h$)')
echo "$LTX" | wc -l                                   # 52

# 1: the refusal surface is FOUR populations, not one. Fail() is a lower bound
for p in 'Fail(' 'Refuse(' 'VT_CHECK(' 'Require('; do
  printf '%-11s %s\n' "$p" "$(echo "$LTX" | xargs grep -oF "$p" | wc -l)"
done                                                  # 375 / 61 / 270 / 53

# A.5: the CLOSED sweep. Case-INSENSITIVE -- the first pass's regex was not,
# and so missed every `NOT ported`. 98 lines, of which 21 are capability sites
echo "$LTX" | xargs grep -niE 'not ported|not implemented|not served|unported' | wc -l

# A.4: the two arms that had already shipped, proved by content not by number
grep -n 'Res2sTwoStageRecipe\|res2s_two_stage' src/vllm/model_executor/models/ltx2_pipeline.cpp
grep -n "DFR's temporal x2/x4 rounds" src/vllm/multimodal/ltx2_video.cpp
git merge-base --is-ancestor 4d7748646 HEAD && echo 'A5 shipped in 4d7748646'
git merge-base --is-ancestor d995c52f0 HEAD && echo 'A6 shipped in d995c52f0'

# A.6: upstream builds no prompt K/V cache, with a positive control
( cd ~/_git/LTX-2 || exit
  for p in kv_cache prompt_kv cached_kv text_kv PromptCache; do
    printf '%-11s %s files\n' "$p" \
      "$(git grep -l "$p" fd4ded7f -- packages/ltx-core packages/ltx-pipelines | wc -l)"
  done                                                # 0 each
  git grep -l attention fd4ded7f -- packages/ltx-core | wc -l )   # 66 -- the control

# Class B registry, and that it still carries exactly these members
sed -n '/^enum class Ltx2UnportedPipelineFeature/,/^};/p' \
  include/vllm/model_executor/models/ltx2_pipeline.h

# A.3: the quantized-compute claim. No PRODUCT consumer; tests are not one
grep -rn 'Ltx2DitQuant' src/ include/ tests/ scripts/ | sed 's/:.*//' | sort | uniq -c
grep -rln 'Ltx2LoadDitFromSafetensors' src/ include/ examples/ | wc -l   # control: 6

# 7: TEN kinds and TWENTY-EIGHT accepted (kind, version) pairs -- not one number
R='/^Ltx2PipelineRecipe ResolveLtx2PipelineRecipe/,/^}/'
awk "$R" src/vllm/model_executor/models/ltx2_pipeline.cpp \
  | grep -oE 'pipeline_kind == "[a-z0-9_]+"' | sort -u | wc -l          # 10
awk "$R" src/vllm/model_executor/models/ltx2_pipeline.cpp \
  | grep -oE 'model_version == "[0-9.]+"' | wc -l                       # 28

# 9: the cadence, as a raw fact. It is NOT multiplied by the row count -- see §9
git log --no-merges --format=%s --since=2026-08-17 HEAD \
  | sed -nE 's/^[a-z]+\(([A-Z0-9][A-Z0-9-]*)\).*/\1/p' | grep -i LTX | sort -u | wc -l

# D3: LTX rows in the quantization matrix, against a positive control
grep -ci 'ltx' .agents/quantization-matrix.md
grep -ci 'nvfp4\|fp8' .agents/quantization-matrix.md

# D4: the missing upsampler checkpoint rows, against a positive control
grep -ci 'upsampl\|upscal' docs/USAGE.md
grep -ci 'ltx' docs/USAGE.md

# D2: the harness never scores adherence. An ABSENCE, so with a control
grep -c adherence scripts/ltx25-render-confirm.sh       # 0
grep -c -- --reference scripts/ltx25-render-confirm.sh  # 4 -- the control
grep -n -- '--adherence-model' scripts/ltx25-render-compare.py | head -1

# D11a: a refusal message calls ported perturbations unported. D11b-D11i below
sed -n '885p' src/vllm/model_executor/models/ltx2.cpp
grep -n 'kSkipA2vCrossAttn' include/vllm/model_executor/models/ltx2_pipeline.h

# 5.1: TI2VidTwoStages landed
git log --oneline --grep='TI2VID-RECIPE'
grep -n 'Ti2VidTwoStageRecipe' src/vllm/model_executor/models/ltx2_pipeline.cpp

# 5.2: the clamp is inside the VIDEO reference item, not the audio one
grep -n '^void Ltx2ConditionVideoByReference\|^void Ltx2ConditionAudioByReference\|std::max(0.0' \
  src/vllm/model_executor/models/ltx2_conditioning.cpp

# Reachability: the family registration and the recipe dispatch
grep -n 'REGISTER_VLLM_VIDEO_FAMILY' src/vllm/multimodal/ltx2_video.cpp
sed -n '2087,2201p' src/vllm/model_executor/models/ltx2_pipeline.cpp
# A24 / A.7: the phase-L6 dtype debt. JOIN LINES FIRST -- the marker is split
# across a line break in ltx2_upsampler.h and a line-based grep drops it silently
for f in $(echo "$LTX" | grep '^include/'); do
  tr '\n' ' ' < "$f" | grep -qE 'phase[ /]*L6|owed with the decoder|owed with the CUDA' \
    && echo "$f"
done | wc -l                                          # 10 of the 52
# ... of which ltx2_loader.h carries it in a `Row:` line and records no dtype debt
grep -n 'phase L6' include/vllm/model_executor/models/ltx2_loader.h   # :5, the Row: line
# ... and of the remaining nine, ltx2.h is the DiT, whose PRODUCTION path already
# has the bf16 arm. That leaves the eight components §A.7 tables
grep -n 'stream_dtype = vt::DType::kBF16' include/vllm/model_executor/models/ltx2_device.h

# A24: upstream constructs all of them in bfloat16, from ONE resolved dtype
( cd ~/_git/LTX-2 || exit
  git show "fd4ded7f:packages/ltx-pipelines/src/ltx_pipelines/distilled.py" \
    | grep -n 'bfloat16\|self\.dtype\|vae_dtype = dtype' )   # :109 sets it; :113..:165 pass it

# A24: the refusal, and that a PRODUCTION path reaches it
grep -n 'RequireF32' src/vllm/model_executor/models/ltx2_text_encoder.cpp   # :54 def, :409 :546
grep -rn 'Ltx2EncodePromptToConditioning(' src/vllm/multimodal/ltx2_video.cpp | grep -v '//'

# 3: the four class-B members OUTSIDE the enum and outside the campaign Out list
grep -n 'kCfgStarRescaling\|kLtxApg\|kLegacyStatefulApg' \
  src/vllm/model_executor/models/ltx2_pipeline.cpp
grep -n 'attn_res_x' src/vllm/model_executor/models/ltx2_video_vae.cpp
grep -ni 'CFGStar\|LtxAPG\|LegacyStateful\|attn_res_x' .agents/specs/ltx-2-5.md   # none
grep -cin 'guider' .agents/specs/ltx-2-5.md                     # 5 -- the control
( cd ~/_git/LTX-2 || exit
  for n in CFGStarRescalingGuider LtxAPGGuider LegacyStatefulAPGGuider; do
    printf '%-24s %s\n' "$n" "$(git grep -n "$n" fd4ded7f -- packages | wc -l)"
  done                                                # 1 each: its own class stmt
  printf '%-24s %s\n' MultiModalGuider \
    "$(git grep -n MultiModalGuider fd4ded7f -- packages | wc -l)"          # 110
  git grep -n 'MultiModalGuider(' fd4ded7f -- packages | wc -l              # 10 constructions
  # attn_res_x: the CALLER passes a kwarg the CALLEE never declares
  git show fd4ded7f:packages/ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py \
    | grep -c attention_head_dim                                            # 1, at :94
  git show fd4ded7f:packages/ltx-core/src/ltx_core/model/video_vae/resnet.py \
    | grep -c attention_head_dim )                                          # 0 -- TypeError

# 6.1 / D11: the population is TWO pattern sets, and neither closes it alone
echo "$LTX" | xargs grep -niE 'not ported|not implemented|not served|unported' | wc -l   # 98
echo "$LTX" | xargs grep -niE '\bowed\b|does not ship|this tree does not|out of this phase' \
  | wc -l                                                                                # 69
echo "$LTX" | xargs grep -niE 'not ported|not implemented|not served|unported|\bowed\b|does not ship|this tree does not|out of this phase' \
  | wc -l                                                                                # 165
# D11b/c/d/e: ONE landing falsifies four statements in four files
git merge-base --is-ancestor d995c52f0 HEAD && echo 'the rounds loop is an ancestor'
grep -n 'requested_temporal_rounds' src/vllm/multimodal/ltx2_video.cpp    # :2275 :2317 :4982 :5020
grep -n 'Ltx2UpsampleVideoLatent(' src/vllm/multimodal/ltx2_video.cpp     # THREE call sites
# D11f/g/h: each contradicts its own translation unit
grep -n 'Ltx2ConditionVideoByGeneratedKeyframeSlots' \
  src/vllm/model_executor/models/ltx2_conditioning.cpp src/vllm/multimodal/ltx2_video.cpp
grep -n 'Ltx2AudioEncoderForward' src/vllm/model_executor/models/ltx2_audio_vae.cpp \
  src/vllm/model_executor/models/ltx2_audio_input.cpp
grep -n 'Ltx2ConvVideoEncode(' src/vllm/model_executor/models/ltx2_video_vae.cpp \
  src/vllm/multimodal/ltx2_video.cpp
# D11i: all four "unported pipelines" are among the ten kinds §7 counts
awk "$R" src/vllm/model_executor/models/ltx2_pipeline.cpp \
  | grep -oE 'pipeline_kind == "(a2vid_two_stage|ti2vid_two_stage|res2s_two_stage|keyframe_interpolation)"' \
  | sort -u | wc -l                                   # 4
```

**§2.1 and §2.2's every figure, from the rule that defines them.** This block IS
the rule: there is no other definition of the count anywhere in the tree, and a
figure this block does not print is not published. Save it and run it.

```python
import glob, re, os
SELF = ".agents/specs/ltx25-completion-scope.md"
HEAD = re.compile(r'^(#{1,6})\s+(.*)$')           # ATX heading, any level
NUM  = re.compile(r'^\d+(\.\d+)*\.?\s+')          # optional "N." / "N.M." numbering
OWED = re.compile(r'^(owed\b|what\s+(is|stays)\s+(still\s+)?owed)', re.I)
SEP  = re.compile(r'^\|[\s:|-]*\|?\s*$')          # a |---|---| table separator
ISSUE = re.compile(r'#(\d{2,6})')
files = [f for f in sorted(set(glob.glob(".agents/specs/ltx25-*.md")) | {".agents/specs/ltx-2-5.md"})
         if os.path.normpath(f) != os.path.normpath(SELF)]
tot = secs = blind = 0; agg = {}; specs = set(); issues = set(); seen = set()
for f in files:
    lines = open(f, encoding='utf-8').read().split("\n"); txt = "\n".join(lines)
    # exactly the two rules `owed_issues()` read the owed section with, and no more
    gate = txt.split("\n## Owed", 1)[1].split("\n## ", 1)[0] if "\n## Owed" in txt else ""
    seen |= set(ISSUE.findall(gate))
    i = 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        if not m: i += 1; continue
        title = NUM.sub("", m.group(2).strip().strip('`*_ ')).strip()
        if not OWED.match(title): i += 1; continue
        secs += 1; specs.add(f); j = i + 1; body = []
        while j < len(lines) and not HEAD.match(lines[j]): body.append(lines[j]); j += 1
        n = 0
        for k, l in enumerate(body):        # item = column-0 bullet, or table DATA row
            if re.match(r'^[*-]\s+\S', l): n += 1
            elif l.startswith('|') and not SEP.match(l):
                if not SEP.match(body[k + 1] if k + 1 < len(body) else ""): n += 1
        seg = "\n".join(body); issues |= set(ISSUE.findall(seg))
        if seg.strip() and seg.strip() not in gate: blind += n
        tot += n; agg[f] = agg.get(f, 0) + n; i = j
print(f"population={len(files)} specs_with_an_owed_section={len(specs)} "
      f"sections={secs} items={tot}")                            # 53 / 49 / 57 / 352
print(f"items_invisible_to_the_gate={blind}")                    # 57
print(f"issues_under_an_owed_heading={len(issues)} "
      f"gate_collects={len(issues & seen)} gate_MISSES={len(issues - seen)}")  # 166 / 159 / 7
print("heaviest:", [(os.path.basename(f), n) for f, n in sorted(agg.items(), key=lambda x: -x[1])[:6]])
```

## Stop conditions

* This row implements no arm. It returns `NEEDS_DECISION`.
* If a reader finds a class-A item that is actually a mirror, that is a finding
  against this spec and it is repaired here, not worked around.

## Owed

* **A ratification, or a refusal, of the GGUF exemption for LTX-2.5.** Owner:
  the developer, through #2526. Five specs assert "not applicable" and no record
  arbitrates. §A.3, D12.
* **A ratification, or a refusal, of the PROMPT-ENHANCER exclusion.** A23. It is
  declared out of scope for every LTX row at `ltx25-res2s-loop.md:271` with no
  citation and no arbitrating record, which is the same unratified-exclusion
  shape §5.3 exists to attack. Owner: the developer, through #2526.
* **A decision on what to fund, and over what period.** §9 derives 27 to 39 rows
  and REFUSES a duration, because the measured cadence was taken on a population
  that differs in kind from the remaining one and because A1 has no mechanism.
  This row will not convert a row count into weeks. Owner: the developer.
* **An owner for A1**, the adherence repair. `ltx25-prompt-adherence.md` states
  that no row exists. This spec does not create one, because the mechanism is
  still in flight on #2520.
* **A row and an issue for the per-stage distilled-LoRA strength**, the one
  residual of the removed A5. It is named out of scope at
  `ltx2_pipeline.cpp:1434` and owned by no row. §A.4.
* **A repair of `scripts/check-agent-record.py`'s owed reader**, D10, with a
  red-first test. It needs its own spec, because it changes checker semantics.
* **The class-D records sweep, D3 through D12.** Each rides the change that
  invalidates it, per AGENTS.md "a record edit rides in the pull request whose
  change made the record stale" -- except D7, D8 and D11, which are stale TODAY
  and which this row's own landing does not repair. Owner: this row, owed.
* **A verdict on the audio VAE's f32 extension.** §A.7. Upstream builds
  `AudioDecoder` in bfloat16 (`distilled.py:156`) and pins float32 only at the
  vocoder (`vocoder.py:575-580`); this tree runs both in f32 on an argued
  extension of the vocoder's contract. It is deliberately outside A24 because it
  owes nothing — it argues — and the sweep that defines A24 reads markers, not
  arguments. Owner: unassigned; it needs a measurement, not a decision.
* **D11's nine statements, and D11b before the other eight.** §6.1 lists them.
  D11b is a user-facing `Fail()` that tells a DFR caller holding the temporal x2
  checkpoint that a shipped arm does not exist, so it costs a user a capability
  rather than a reader a fact. D11b, D11c, D11d and D11e are one landing's
  fallout (`d995c52f0`, [#986](https://github.com/mudler/vllm.cpp/issues/986)) and
  repair together. Owner: this row, owed.
* **A second scored render.** D1. Owner: `LTX25-PROMPT-ADHERENCE`.

## Now

`LTX25-COMPLETION-SCOPE` is `DONE` as a survey and returns `NEEDS_DECISION` on
what to fund, on the two unratified exclusions, and on the schedule. The
inventory is §3 through §6, the sequence is §8, the count is §9, and what could
not be determined is §10. **Every number is produced by a command in `## Gates`;
the ones that were not have been withdrawn.**

Three findings a reader should carry away. **Class C is empty:** nothing in
LTX-2.5 is externally blocked on the engine side, and the three arms recorded as
unreachable are ordinary work — one of them already landed. **The survey's own
method was weakest where it read this tree**: two class-A entries described arms
that had shipped, one described a cache upstream never builds, one pointed at a
refusal for a different capability, and a closed sweep over the refusal surface
found nine gaps in no class at all. And **every closure this row claimed over its
own tree has since failed** — three times now, each time in the same direction.
The refusal surface was a lower bound on the capability surface (A24, §A.7); "one
records defect" was nine (§6.1); "four correct mirrors" was eight (§3). The rule
that survives is the one §2.1 states about counts and now applies to
populations: **a sweep is a property of its patterns, so the patterns ship with
the population or the population does not ship.**

What actually blocks "complete" is still a failing correctness gate with no owner
and no mechanism.
