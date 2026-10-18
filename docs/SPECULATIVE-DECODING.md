# Speculative decoding

A draft proposes tokens, the full model verifies them in the same step, and an
accepted proposal skips a full model step. It is **opt-in and off by default**;
with no speculative config the engine is byte-identical to the non-speculative
one.

Selection mirrors vLLM's own CLI: `--speculative-config '<json>'`, taking the
same JSON object vLLM takes, so a config written for vLLM works here.

## Methods

The engine accepts five `method` strings. Anything else is refused at load with
the list of accepted ones (`src/vllm/config/speculative.cpp`). A `method` selects
exactly one speculator; the [drafter chain](#the-drafter-chain-and-why-it-does-not-run-yet)
below is the one place a document can name several, and it does not run yet.

| `method` | Draft | k | State |
|---|---|---|---|
| `mtp` | the target's own `mtp.*` head | 1 | Gated, and the default recommendation. Token-exact against vLLM at concurrency 1, ~1.04x its speculative-on decode |
| `dflash` | a separate z-lab block-diffusion checkpoint | block size, e.g. 16 | Gated. 2.9x over speculative-off at concurrency 1, at or above vLLM's own DFlash-on |
| `dspark` | a separate DSpark checkpoint | block size, at least the draft's own | Runs end to end, **not gated**. Token-identical to speculative-off on the 35B. Speculation works (drafts proposed and accepted), but the cross-engine ratio is UNSETTLED: the last matched-and-warm paired measurement is 0.834x of the pinned oracle. No speed win is claimed |
| `ngram` | none, drafts come from the prompt's own suffix | required, no default | Accepted and wired; no published measurement |
| `draft_model` | a separate full model | from the draft | **Config only.** The JSON parses, but the engine has no branch for it and refuses the load |

MTP and DFlash are the two with binding numbers behind them; the per-method
detail below and in [the benchmark record](BENCHMARKS.md) says which is which.

## Which keys the JSON accepts

The object is checked key by key and **nothing is dropped**. A name this engine
cannot honour is refused at startup, by name, before a byte of weights is read.
Until [#1160](https://github.com/mudler/vllm.cpp/issues/1160) an unrecognised key
was silently discarded, so `"draft_sample_method":"probabilistic"` started a
server that drafted greedily and reported nothing, and a typo such as
`"num_speculatve_tokens"` quietly took the resolved default instead of the value
that was typed.

| Key | Accepted |
|---|---|
| `method` | `mtp`, `dflash`, `dspark`, `ngram`, `draft_model` |
| `num_speculative_tokens` | a positive integer, or absent for the method's own default |
| `model` | the draft checkpoint path or HF repo id |
| `prompt_lookup_min`, `prompt_lookup_max` | an integer of at least 1, `ngram` only |
| `draft_sample_method` | `greedy` only, which is upstream's default and what this engine does |
| `rejection_sample_method` | `standard` only, which is upstream's default and what this engine does |
| `vllm_cpp` | this engine's own extension object; its only key is `drafter_chain`, described below |

There are two kinds of refusal, worded differently on purpose. A name vLLM's own
`SpeculativeConfig` declares, such as `quantization` or `max_model_len`, is
reported as a real vLLM field this engine does not implement. Any other name is
reported as unknown, together with the list above, because that one is usually a
typo. `draft_sample_method: probabilistic` and the `synthetic` and `block`
acceptance variants name row `SPEC-ACCEPT-VARIANTS`, which owes them.

## The drafter chain, and why it does not run yet

`--speculative-config` also accepts one key vLLM does not have. It lives under a
`vllm_cpp` object so that it can never be confused with a vLLM field, which is
the same place `--offload-config` keeps this engine's own residency keys:

```json
{"vllm_cpp": {"drafter_chain": [
  {"method": "dflash", "model": "z-lab/Qwen3.6-27B-DFlash", "num_speculative_tokens": 16},
  {"method": "ngram", "num_speculative_tokens": 4}
]}}
```

It describes a **preference order**, not an ensemble: try the first speculator,
and where it produces no draft for a sequence, try the next. One speculator wins
per sequence, nothing merges proposals, and the verify is untouched.

**The engine refuses this document today, by name, before it reads a byte of
weights.** The field is parsed, validated and stored; nothing resolves it yet.
Refusing is the point rather than an oversight: a chain that parsed and was then
ignored would start a server drafting with one speculator, or with none, under a
document whose author asked for several, and any measurement taken there would
describe a configuration nobody chose. Resolution is owed by row
`SPEC-DRAFTER-CHAIN`, [#1522](https://github.com/mudler/vllm.cpp/issues/1522).

Everything about the field except the running is real, so a document can be
written and checked now:

| Rule | Refused |
|---|---|
| `method` at the top level, beside a chain | yes, they are mutually exclusive: name every speculator as an entry |
| `model`, `num_speculative_tokens`, `prompt_lookup_min`, `prompt_lookup_max` at the top level, beside a chain | yes, each configures one speculator and there is no entry it belongs to |
| `draft_sample_method`, `rejection_sample_method` at the TOP LEVEL, beside a chain | no, both describe the verify and the draft sampling rule, which are engine-wide |
| either of those two INSIDE a chain entry | yes, by name (#1598): they are engine-wide rather than per drafter, and the message says the engine honours them and where to spell them |
| an entry naming anything but `mtp`, `dflash`, `dspark` or `ngram` | yes, by name, including a real vLLM method such as `eagle3`, and including `draft_model`, whose chain arm is owed |
| an entry missing the key its method requires | yes, by the same rules the top-level `method` follows |
| any other entry key the top level does not honour | yes, by name, split the way #1160 splits the top-level document: a vLLM `SpeculativeConfig` field this engine does not implement is refused as that, and a name nobody declares is refused as unknown with the accepted list |
| the same method named twice | yes: per-drafter attribution keys on the method name, so two entries of one method cannot be told apart in the counters |
| an empty chain, or a `vllm_cpp` object with no `drafter_chain` | yes: omitting `vllm_cpp` is how a document says "no chain" |
| a chain of one entry | no, that is the degenerate preference list and it is legal |

**With no `vllm_cpp` key, nothing above applies.** Every document that worked
before this field existed keeps its exact meaning, key for key, which is what
makes the field additive rather than a fork of vLLM's surface.

That is a claim about MEANING, and it is deliberately not a claim of
byte-identity. There is exactly one visible delta on a chain-free document, it
was measured, and it is stated here rather than left for a reader to find: the
accepted-key list quoted at the end of an unknown-key or unimplemented-key
refusal now also names `vllm_cpp.drafter_chain`. Every such refusal message is
therefore longer by that tail. The change is required rather than incidental —
an accepted-key list that omits an accepted key stops closing the user's search,
which is the whole reason #1160 appends one. No document's parse result moves,
no key changes meaning, and no error changes which guard produced it.

The distinction matters beyond ergonomics. Draft sampling and verify are greedy
here, so a dropped `probabilistic` produced a **deterministic** run when a sampled
draft was asked for, and a deterministic run is adjudicable by the token-exact
greedy gate while the configuration the user actually requested is not. A
silently downgraded flag therefore lets a parity or benchmark number be taken
under a configuration nobody chose.

## MTP

- **Models:** the Qwen3.5 / 3.6 gate checkpoints that ship an `mtp.*` draft head
  in their safetensors (Qwen3.6-27B and Qwen3.6-35B-A3B). Both are GDN hybrids,
  and the speculative path is wired through the linear-attention (GDN) recurrence
  and short causal convolution as well as the attention layers.
- **Depth is configurable.** `num_speculative_tokens` sets how many tokens the
  MTP head drafts per step, by looping the single head autoregressively. It
  defaults to the checkpoint's `mtp_num_hidden_layers`, which is 1 on both gate
  checkpoints, so the default is unchanged. A value above `n_predict` must be a
  multiple of it, mirroring vLLM. Depth is a pure throughput lever: greedy
  decoding plus accept-if-equal rejection makes the emitted tokens identical at
  every k, which is proven on CPU for k=1..4 against speculative-off. That
  identity is exactly why a token gate cannot see a clamped depth, so each CPU arm
  asserts TWO positive witnesses instead, because neither one alone is enough. The
  DRAFT DECODE FORWARDS the propose ran, `k-1` per propose call, catch a propose
  that never entered the loop. They do NOT catch padding: a propose that runs
  every forward and then overwrites all k columns with its step-0 draft reports
  `k-1` truthfully. The second witness counts the propose calls whose DELIVERED
  draft row was not a pure function of its own first column, which is what a
  padded row is, and it is measured non-zero at every k >= 2. Neither witness
  proves PER-COLUMN PROVENANCE, that column j came from forward j. What the
  CPU tier does NOT show is acceptance at depth: no draft is accepted at any depth
  on the synthetic gate model, so the accept path at k>1 is unexercised there. A
  non-zero acceptance COUNT at depth would not close provenance either, because a
  padded row is accepted at column 1 whenever the target's own greedy continuation
  repeats a token, which real text does routinely. Closing it needs an acceptance
  RATE measured against a PADDED CONTROL on the same workload. The cross-engine
  speed comparison at k>1 and the DGX three-way at k=2..4, which must run that
  control, are still owed
  ([#81](https://github.com/mudler/vllm.cpp/issues/81) M1/M2), as are
  batch-size-keyed dynamic depth and acceptance-driven adaptive depth.
- **Correctness:** at concurrency 1 the speculative-on greedy output is
  token-for-token identical to both the speculative-off output and vLLM's own MTP
  speculative greedy output on the same prompt.
- **Speed:** measured about 1.04x faster than vLLM's own speculative-on decode at
  concurrency 1 (see [Measured result](#measured-result)).

## DFlash (block diffusion)

DFlash drafts a whole block in one non-autoregressive pass from a separate
z-lab draft checkpoint over the same target. `num_speculative_tokens` is
required and is the draft's block size, e.g. 16.

```bash
server --model /models/Qwen3.6-27B \
  --speculative-config '{"method":"dflash","model":"<draft-path>","num_speculative_tokens":16}'
```

Gated on the 27B: end-to-end output matches the vLLM DFlash-on golden under the
ratified near-tie rule, and at concurrency 1 it is 2.9x our own speculative-off
throughput and at or above vLLM's own DFlash-on decode. Speculative-off stays
byte-identical.

### DFlash2 checkpoints

DFlash2 adds a grouped dynamic depthwise convolution and a candidate selector
that replaces DFlash1's per-slot argmax with a scored path walk over the target
head's top-K. **A safetensors checkpoint that declares `DFlash2DraftModel` now
drafts end to end**: it loads, runs the convolution around every attention and
MLP sublayer, scores the selector's transition lattice, and walks that lattice
from the verified anchor to produce the block's tokens. Use it exactly as a
DFlash1 draft:

```bash
server --model /models/Qwen3.8-27B \
  --speculative-config '{"method":"dflash","model":"<dflash2-draft-path>","num_speculative_tokens":7}'
```

Startup prints a notice naming what runs and what is still owed. The engine
never substitutes the DFlash1 per-slot argmax for a DFlash2 block: that would
emit valid target tokens while silently reducing draft acceptance, which is the
failure this lane is built to avoid because no token gate can see it. If the
walk cannot run, the draft is refused by name instead.

Two limits are worth knowing before you use it. **Draft sampling is greedy
only** -- `draft_sample_method: "probabilistic"` is refused by name here, as it
is for every method, so DFlash2's noised walk and its cached proposal
distribution are not built. And **no DFlash2 throughput number has been taken
yet**: correctness comes first on this lane, and a DFlash2 draft additionally
runs its block forward off the paged CUDA-graph fast path, because the candidate
selector needs the hidden states of the same forward its logits came from.

**A GGUF DFlash2 checkpoint now drafts too**, in all three published arms:
bf16, Q8_0 and Q4_K_M. Point `--speculative-config` at the `.gguf` file exactly
as you would at a safetensors directory. The file is identified from
DFlash2-specific metadata rather than from an architecture string, because a
GGUF declares none and the published DFlash2 GGUF writes
`general.architecture = dflash`, the same value a DFlash1 drafter writes.
Ordinary DFlash1 GGUF drafts are unchanged.

What that claim rests on, so you can weigh it: the three ENCODINGS are gated by a
synthetic draft this engine writes itself in each block format and then drafts
from, value-for-value against the suite's own encoders, and the published files
are read for their NAMES, SHAPES and TYPES. **The Q4_K_M arm has additionally
been loaded for real**, on 2026-08-21: the published 1.06 GiB file was pointed at
a 27B target through the command-line client, mapped and decoded every tensor at
the full geometry, proposed seven speculative blocks and generated. The BF16 and
Q8_0 arms are still gated by the synthetic draft and the header read alone.
`Q4_K_M` is also llama.cpp's usual mixture rather than one encoding: the
published file is 32 F32, 45 Q4_K and 4 Q6_K tensors
(`blk.{2,4}.{attn_v,ffn_down}.weight`), all of which this loader decodes.

One property of this container is worth knowing before you pick an arm: **a GGUF
drafter is dequantized to bf16 at load.** That is this lane's design rather than
a fallback -- a DFlash draft is a handful of layers, so the loader hands the
shared weight path bf16 views and reuses the whole safetensors body -- but it
means a Q4_K_M draft costs its bf16 residency and not its file size. Summed
over the published 27B drafter's own tensor table, that is 3 848 808 960 bytes
(3.584 GiB) of draft weights whichever of the three files you load -- against
1.06 GiB on disk for the Q4_K_M one -- of which 254 279 680 bytes are the
candidate selector's two codebooks, which the DFlash1 lane never allocates at
all. Pick the smallest file to save disk and download time; it will not save
memory.

**The TARGET's `lm_head` may be quantized, as long as this engine can compute
with it.** A DFlash or DFlash2 draft owns no output head; it runs the target's.
Until [#1628](https://github.com/mudler/vllm.cpp/issues/1628) that head had to be
stored as dense bf16, so pointing a draft at a safetensors target whose
`lm_head.weight` is ModelOpt or compressed-tensors NVFP4 refused the load with
`dflash: target tensor lm_head.weight is not BF16 (got U8)`. Such a head is now
kept PACKED and multiplied by the same W4A16 GEMM the target's own logits take,
so the candidate selector reads the target's exact top-K rather than a widened
head's.

What is still refused, and why the distinction is the point: a head this engine
would have to WIDEN to read stays refused by name, because the selector's whole
input is the target head's exact top-K and a widened head produces a different
candidate set with no visible symptom -- the verify is lossless, the emitted
tokens are still the target's, and only acceptance falls. That covers a GGUF
target's `output.weight` (dequantized on the way in), an FP8 `lm_head`, and an
NVFP4 head with the activation divisor in force under `VT_MODELOPT_W4A4=1`. A
DSpark draft still refuses every quantized target head. The DFlash1 lane is
unaffected either way.

**This change's own gates are CPU.** The packed-head equality above is
measured on CPU fixtures, and the CUDA arm and a real NVFP4 target are owed a
measurement ([#1628](https://github.com/mudler/vllm.cpp/issues/1628)).

**The acceptance gate has now passed, and no speed result has been taken.**
Measured 2026-08-21 on a GB10 against vLLM built at
[vllm#52816](https://github.com/vllm-project/vllm/pull/52816) head `66e5414c6`,
`Qwen/Qwen3.8-27B` bf16 with the `z-lab/Qwen3.8-27B-DFlash2` drafter, k=7,
greedy, one request at a time, 64 tokens on each of four prompts:

| | |
|---|---|
| output tokens | **4 of 4 prompts identical to vLLM's** |
| draft tokens | **45 of 47 blocks byte-identical to vLLM's own drafts** |
| acceptance | **identical per prompt** -- 49, 54, 54, 52 on both engines |

The two draft blocks that differ each differ by one token, in a part of the
selector that is a floating-point reduction and is specified within an envelope
rather than bit-exact; both blocks then produced the same output tokens.

**That head is one merge behind vLLM's `main`, and the numbers above are still
the ones that were measured.** vllm#52816 MERGED on 2026-08-21 at 05:27:22Z, at
head `3406ec1d` and merge commit `b389ac29` -- 46 minutes before this row's W6
work commit `bb416e0ae` was authored at `06:13:50Z`. **The 46 minutes is the
interval between those two COMMITS, and it is not the gap to the run.** The
wheel above was built at head `66e5414c`, which is an earlier head of the same
pull request than the merged `3406ec1d`. WHEN it was built and run is NOT a
number this document can state, because no timestamp for that build or that run
survives anywhere in the tree -- not in the golden, not in the record -- so
where the run falls against the merge instant is unmeasured, and transferring
the commit interval onto it would be a measurement nobody took. The measurement
stays pinned to `66e5414c` because that is what executed, and re-labelling a run
with a head it never ran would be a false pin. Re-reading the gates at the
merged head is owed under
[#1561](https://github.com/mudler/vllm.cpp/issues/1561).

**The backend that comparison ran on is not vLLM's default here.** vLLM was run
on `TRITON_ATTN` rather than on the flash-attention backend it auto-selects,
which is a deliberate constraint recorded in
[#1456](https://github.com/mudler/vllm.cpp/issues/1456) and is stated because it
is not free: running the same vLLM on its two backends changes its own answer on
one of these four prompts and moves its own acceptance by six points.

**Still no throughput number**, and two things would bound one taken today. A
DFlash2 draft here runs off the paged CUDA-graph fast path, because the candidate
selector needs the hidden states of the same forward its logits came from, while
vLLM captures a CUDA graph for its DFlash2 draft step. And every wall-clock
figure above was taken with a 51.75 GiB checkpoint read over a network mount, so
it measures loading rather than decoding
([#1314](https://github.com/mudler/vllm.cpp/issues/1314)).

**The exact checkpoints this was built and gated against.** Every sha256 below
was computed locally from the downloaded artifact. The HuggingFace tree API
returns no usable `lfs.oid` for these files, so a hub-reported hash is not
evidence here.

| Arm | Repo and revision | File | Bytes | sha256 |
|---|---|---|---|---|
| Draft, bf16 safetensors — DRAFTS | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | `model.safetensors` | 3 848 817 896 | `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c` |
| Draft, GGUF — DRAFTS | `z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `57ab3265056d4024870b0621cfc2c127537020ed` | `Qwen3.8-27B-DFlash2-BF16.gguf` | 3 860 293 152 | `26af33a15b21475d668e4ee55639beea49932e7360b1144c6282721bcd127c14` |
| Draft, GGUF — DRAFTS | same | `Qwen3.8-27B-DFlash2-Q8_0.gguf` | 2 056 414 752 | `7f1c9a31a6ed40044c69f6508b50fd63b87abd8e1fb7fe4290303df549153751` |
| Draft, GGUF — DRAFTS | same | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1 143 006 752 | `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594` |
| Draft, EXL3 safetensors — LOADS | `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` @ `4f0436269bca761b071f05319e8e04a87cc633f9` | `model.safetensors` | 1 470 916 078 | `6b2e3afc694a343b7f3f0edfe5925e460762fc9ede4699165b577ca0733c8e56` |
| Draft, ModelOpt NVFP4 safetensors — LOADS | `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal` @ `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1` | `model.safetensors` | 1 550 153 248 | `2228b9b22e93a88d84556419c879448ab6c490ae65c4c0b166f4962190ddbf26` |
| Target the draft heads | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | published shards | — | — |

**The EXL3 draft row says LOADS and not DRAFTS, and the difference is exact.**
That file was read by `vllm::LoadQwen3DFlash`, the function
`--speculative-config` reaches, and all 36 of its quantized modules came back at
bits 5 with the `mul1` codebook and the geometry its `config.json` declares
([#2495](https://github.com/mudler/vllm.cpp/issues/2495) item 7,
[spec](../.agents/specs/model-dflash2-exl3.md)). No generation against a real
target has been run with it and no throughput number is claimed. Its published
partner target, `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`, stores an EXL3 `lm_head`
that `SharedHeadSource::LoadInto` does not yet read — items 5 and 6 of the same
issue — so today this draft has to be paired with a target whose head this
engine already reads. That target also carries 137 modules at 3 bpw, and `(3, 2)`
is not in the CUDA instantiation list, so the published pair cannot run on a CUDA
queue at all ([#2574](https://github.com/mudler/vllm.cpp/issues/2574)). This
draft's own 36 modules are bits 5, which is instantiated; it is the target half
that refuses.

**The NVFP4 draft row says LOADS for the same reason, and one more.** Its arm is
the DECLARATION: the loader reads the DRAFT's own `quantization_config`, as
upstream's `get_draft_quant_config` does
(`vllm/model_executor/models/utils.py:929-948` at the parity pin), honours
`exclude_modules` / `ignore` module by module, and cross-checks the declaration
against the shipped tensors in both directions. All 35 of its quantized modules
come back packed with the geometry its `config.json` declares
([#2758](https://github.com/mudler/vllm.cpp/issues/2758),
[spec](../.agents/specs/model-dflash2-nvfp4.md)). Four modules have NO packed
owner here -- `fc`, `candidate_selector.hidden_projection` and both conv
`kernel_projection`s -- and this artifact excludes all four; a draft that
quantized any of them is refused by name naming the module and the missing
owner. **No device has run it.** `vt::MatmulNvfp4` is CUDA-only, so the gate is
a load gate and the forward branch that binds the packed method has never
executed; leg F of `BENCH-QWEN38-27B-SOTA`
([#2761](https://github.com/mudler/vllm.cpp/issues/2761)) is the leg that would.
It also declares `quant_algo: "NVFP4"` -- W4A4 -- on all 35 and this build runs
them W4A16, which the load prints one line about
([#2760](https://github.com/mudler/vllm.cpp/issues/2760)).

A repo id alone is not a pin, because checkpoints get re-quantized in place
under an unchanged name. The revisions above are what these results describe.

## n-gram

The proposer matches the sequence's own suffix to draft tokens. It needs no
draft checkpoint. `num_speculative_tokens` is required. Both lookup bounds
default to 5.

```bash
build/examples/vllm-server --model /models/Qwen3.6-27B \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4}'
```

The engine verifies these drafts through the shared speculative decoding path.

### Kolibri-1 on CPU

Kolibri-1 accepts n-gram speculation through the CLI and server. Use the
[Kolibri-1 checkpoint](../.agents/specs/kolibri-1-cpu.md#the-checkpoint)
at your local model path. This example uses the configuration from the recorded
CPU comparison: up to 5 draft tokens and lookup lengths from 1 to 4.

```bash
build/examples/vllm-cli --model /models/Aleph-Alpha/Kolibri-1 --device cpu \
  --prompt "The capital of France is" --max-tokens 32 --temperature 0 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":1,"prompt_lookup_max":4}'
```

Omit `--speculative-config` to disable speculation. The server accepts the same
JSON through `--speculative-config`. Set `temperature` to `0` in each request
for greedy decoding. N-gram speculation uses the synchronous scheduler.

The [recorded engine comparison](../.agents/specs/kolibri-1-cpu.md#now)
found identical tokens with speculation enabled and disabled on nine prompts,
run one at a time. This preserves the existing comparison with oracle outputs,
including its adjudicated near-tie differences. It does not establish strict
oracle token parity on every prompt.

The [cache and engine tests](../tests/vllm/models/test_kolibri1_ngram_spec.cpp)
cover the cache layout without weights and compare generated tokens when the
checkpoint is available. Acceptance at batch sizes above one and on real
traffic remains unverified. Low draft acceptance can increase verification
work, so these settings are not a speed guarantee for other prompts.

## DSpark (semi-autoregressive block drafting), in progress

DSpark drafts a whole block in one parallel pass and then adds intra-block
dependency with a small sequential head, so a block draft stops being k
conditionally independent guesses. It is the DFlash drafter plus a low-rank
Markov transition bias.

Current state (`SPEC-DSPARK`): the config, the Markov head and draft model, both
published checkpoint layouts (native `deepseek-ai/dspark_qwen3_*` and
Speculators `RedHatAI/*.dspark`), the sequential sampler and the runner wiring
have landed, and the path runs end to end on the Qwen3.6-35B-A3B gate model.
DSpark now genuinely speculates on the Qwen3.6-35B-A3B gate model: draft tokens
are proposed and accepted, and throughput went from 6.78 to 41.89 tok/s once an
engine-wide bug was fixed that had been silently discarding EVERY speculator's
drafts on the CLI and server path (`check_for_draft_tokens` was never threaded
into `EngineCoreProc`, so `post_step` returned before installing anything).

On that model its greedy output is **token-identical to speculative-off across
all 48 tokens and reproducible** (both arms run on the synchronous path; the
speculative path forces async scheduling off, so a fair comparison has to force
it off on the other side too).

**It is still not gated**, and the cross-engine number is UNSETTLED rather than
merely owed. The host-side sequential stage described above was since moved on
device, the T=1+k verify was captured, and the draft chain made sync-free; all
three are byte-identical and landed. What is not settled is the ratio against
vLLM's own DSpark. Ratios recorded through 2026-08-13 (0.957x-0.989x) were taken
with a SINGLE COLD oracle invocation per paired run, so the denominator paid
compile-JIT the numerator did not; with the oracle warm and generation length
matched, the paired measurement is **0.834x**. Those two cannot be differenced
directly, because the gate host was reimaged in between and is no longer the same
machine. The deciding experiment is a single cold oracle invocation on the
current box. The [speculative-decoding benchmark page](benchmarks/speculative-decoding.md)
records the pending run. Until it runs, no speed claim in either direction is
supportable. The acceptance-rate band and the other target families remain
owed. The loader refuses a GGUF target or a target architecture without an
auxiliary multi-tap by name.

```bash
main --model /models/Qwen3-4B \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","num_speculative_tokens":7}'
```

`num_speculative_tokens` is required for a DSpark draft (a native DSpark config
carries no `n_predict`), and it must be at least the checkpoint's block size,
a smaller value produces incorrect output rather than merely lower acceptance,
so it is rejected.

### Which DSpark draft the loader will take

The lane is decided by the DRAFT's own `config.json`, not by the method string
you typed. Three architecture spellings route to the Qwen3 DSpark lane:

| The draft declares | Routes to |
|---|---|
| `architectures: ["Qwen3DSparkModel"]` | the Qwen3 DSpark lane |
| `architectures: ["Gemma4DSparkModel"]` | the Qwen3 DSpark lane (same loader) |
| `architectures: ["DSparkDraftModel"]` with `model_type: "qwen3"` | the Qwen3 DSpark lane |

The third row is vLLM PR
[52197](https://github.com/vllm-project/vllm/pull/52197), merged 2026-08-17,
which is AHEAD of the pinned oracle; it is mirrored here because the pinned
behavior is wrong for a checkpoint that is already published
(`RadixArk/Qwen3.8-27B-DSpark` declares exactly that pair).

A draft config that carries no `architectures` key at all is not classified. It
loads exactly as it did before, because an absent key is not evidence of a lane.

Anything else that names an architecture is the DeepSeek-V4 DSpark draft, whose weights ship inside the
DeepSeek-V4 target rather than in a separate checkpoint. vLLM rewrites such a
config onto `model_type: "deepseek_v4"` and loads it; **this engine refuses it by
name** instead, because it carries only a stub for that model and the lane needs
two DGX Sparks. A refusal that names the missing arm is what you get, rather than
a load that fails later on a missing key.

## The flag

On the OpenAI server:

```bash
server --model /models/Qwen3.6-27B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

On the example CLI:

```bash
vllm-cli --model /models/Qwen3.6-27B \
  --prompt "The capital of France is" --max-tokens 64 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Over the C ABI (`include/vllm.h`, ABI v6):

```c
vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/models/Qwen3.6-27B";
mp.speculative_config = "{\"method\":\"mtp\",\"num_speculative_tokens\":1}";
vllm_engine *engine = NULL;
vllm_engine_load(&mp, &engine);   /* NULL/"" speculative_config => no speculation */
```

The JSON is parsed into the same `vllm::SpeculativeConfig` the C++ API takes
programmatically (`EngineParams::speculative_config`). The examples above use
`mtp`, where `num_speculative_tokens` defaults to the checkpoint's head depth.
See [Methods](#methods) for the other spellings and what each one requires. A malformed document, an unsupported
method, a missing `num_speculative_tokens` where the method needs one, or a
checkpoint with no `mtp.*` head fails the load loudly at startup rather than
running silently without speculation.

## Measured result

Concurrency 1, Qwen3.6-27B (NVFP4) on GB10, our speculative-on decode versus vLLM
0.25.0 running the same speculative config:

| Metric | Ours (spec on) | vLLM (spec on) |
|---|---:|---:|
| Time per output token, prose | 66.2 ms | 69.1 ms |
| Time per output token, code | 62.95 ms | 65.3 ms |
| Decode throughput, prose | 15.1 tok/s | 14.4 tok/s |
| Decode throughput, code | 15.7 tok/s | 15.1 tok/s |
| Drafter acceptance rate | 0.85 prose / 0.92 code | 0.84 |

Speculation helps both engines about 1.5 to 1.6x at this operating point, and our
engine is already about 4% faster than vLLM with speculation off, so the lead is
preserved with it on. The extra state speculation needs (a doubled recurrent-state
slot at k=1, plus the draft cache and head) costs about 3.6 GB, well inside the
box's unified memory. The [speculative-decoding benchmark page](benchmarks/speculative-decoding.md)
carries every speculator's current published position. The higher-concurrency legs of this
particular A/B are not among them and stay owed.

## Concurrency above 1

The concurrent (multi-request) path, where a speculative request shares a
scheduler step with an ordinary prefill, is implemented. The GDN layer splits the
batch into its speculative and ordinary rows, runs each through its own recurrence
over disjoint per-request state, and merges the results back; because the state is
disjoint, the mixed batch's output equals the two requests run alone, bit-for-bit,
at the real 27B and 35B dimensions. Speculation keeps helping at concurrency 2, 4
and 8 on the 27B (about 1.5x our own speculative-off throughput).

## Limitations

Each of these names the method it applies to.

- **MTP depth above 1 has no speed number yet.** The multi-step propose is
  built and token-exactness at k=1..4 is proven on CPU, but the DGX three-way at
  k=2..4 on the 27B and 35B and the cross-engine throughput A/B at matched k are
  both owed ([#81](https://github.com/mudler/vllm.cpp/issues/81) M1/M2), so no
  speed claim is made for k>1. Batch-size-keyed dynamic depth and
  acceptance-driven adaptive depth are unbuilt. A step whose actual draft count
  differs from the configured k (the scheduler clamps drafts to the step's token
  budget) falls out of the captured verify graph silently,
  [#1020](https://github.com/mudler/vllm.cpp/issues/1020).
- **EAGLE, EAGLE3, Medusa and the rest are not wired.** Of vLLM's thirteen
  `SpeculativeMethod` strings the engine accepts five, and `draft_model` among
  those is config-only. The remainder are inventoried, not implemented.
- **DSpark is not gated.** It runs end to end and is token-identical to
  speculative-off on the 35B, but it is about 2% slower than plain decode at
  concurrency 1, and the comparison against vLLM's own DSpark, the
  acceptance-rate band and the other target families are all still owed.
- **MTP needs Qwen3.5/3.6 with an `mtp.*` head.** The target may be a safetensors
  directory or a `.gguf` converted WITH the head (llama.cpp's layer-indexed
  `nextn` block); a GGUF exported `--no-mtp` is refused and says so.
- **On an NVFP4 safetensors target the concurrency-1 identity above is not
  currently reliable.** Measured on the 35B A3B NVFP4 safetensors: its logits
  land on a coarse grid that yields EXACT ties between distinct tokens, and at
  such a tie speculative-on and speculative-off, and even two speculative-off
  runs, can pick differently. The same weights loaded from GGUF, which expands
  to bf16, show no such ties and are token-identical. Open; see
  [the current project status](../README.md#project-status).
- **Concurrency above 1 is not token-stable for the 27B.** Its greedy output is
  not bit-stable across batch shapes even with speculation off (changing the batch
  size flips a few near-tie tokens), so exact token-for-token agreement between
  speculative-on and speculative-off is a concurrency-1 property only. Correctness
  above concurrency 1 rests on the model-independent bit-exact GDN split-merge
  proof and on matching drafter acceptance, not on identical token streams.

## Consuming it programmatically

The flag is a thin wrapper over the library surface:

```cpp
vllm::entrypoints::EngineParams params;
params.speculative_config =
    vllm::ParseSpeculativeConfigJson(R"({"method":"mtp","num_speculative_tokens":1})");
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, params);
```

Leaving `speculative_config` unset (the default) loads the ordinary engine, which
is byte-identical to a build with no speculative code compiled in.
