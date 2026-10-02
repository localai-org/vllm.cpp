# Muse Glimmer

Muse Glimmer is a 30B multimodal model. `MuseGlimmerForCausalLM` and
`MuseGlimmerForConditionalGeneration` both forward, and the perception encoder is
wired, so an image or video prompt runs instead of being refused.

**What has been measured is much narrower than "it works."** Read
[what has actually been checked](#what-has-actually-been-checked) before you rely
on any of it. Nothing has run end to end through the server, and no speed number
exists against vLLM (a secondary llama.cpp bar exists, #333).

## Run the text tower from a GGUF

The text tower loads from a `muse-glimmer`-architecture GGUF, so the 30B model
runs from a roughly 17 GB k-quant instead of a roughly 60 GB BF16 checkpoint.
Point `--model` straight at the file. The config comes from the GGUF's own
metadata, so no `config.json` is needed:

```sh
./build/examples/vllm-server --model /path/to/muse-glimmer-30B-kquant-17gb.gguf
```

Both published k-quants load: `muse-glimmer-30B-kquant-17gb.gguf` and the mixed
per-tensor `muse-glimmer-30B-kquant-dynamic.gguf`.

The standard GGUF residency knobs apply, which are `VT_GGUF_KEEP_QUANT`,
`VT_GGUF_MMAP`, and `VT_CPU_REF`. `o_proj`, the attention output gate,
`down_proj`, and the merged `gate_up` stay quantized. The merged QKV, `lm_head`,
and the embedding table expand to BF16, because the shared forward consumes them
in a form a block encoding cannot take.

**Image and video need the BF16 safetensors.** The released
`mmproj-kquant.gguf` ships its patch embedding without the `patch_temporal`
axis, so half the weight is not in the file. Loading it is refused by name.

### The GGUF k-quant against llama.cpp: token gate, 16/16 inside the near-tie band

`tests/parity/test_muse_glimmer_gguf_paged_engine.cpp` drives the standard
16-prompt battery, 32 greedy tokens each, through `LoadedEngine::FromModelDir`
on `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (sha256 `4cc57c0f...f60e`), CPU,
and compares with the registered llama.cpp oracle at its pin `b10451` on the
same file (`scripts/llamacpp/llamacpp_oracle.cpp`, 2026-09-30). 10 of 16
prompts are token-exact over all 32 tokens. The other 6 differ from the
oracle's free-running greedy, and at every one of their cells the oracle's own
teacher-forced gap on our prefix is at most **93 mnats** (the band is 500), so
there are zero forward-divergent cells. Several of those cells have gap 0: our
token is llama.cpp's own argmax on our prefix, and the difference is between
llama.cpp's batched and single-token numerics. The earlier record ("agrees on
the first token, then diverges") predates this gate and is superseded.

This is quantization-matched agreement with llama.cpp, not a bf16 claim and
not a vLLM result. The golden is `tests/parity/goldens/muse_glimmer_30b_q4km/`.

Two defects had to be fixed to get that far: the GGUF tokenizer gap
([#347](https://github.com/mudler/vllm.cpp/issues/347), where `pre llama4` is the
GPT-4o and o200k family) and the converter's Q and K RoPE row permutation
([#359](https://github.com/mudler/vllm.cpp/issues/359), which produced
`" is is is ..."`).

### An absent config key takes the architecture's value

A key that a config or a GGUF omits falls back to Muse Glimmer's own constant,
not to a neutral one ([#412](https://github.com/mudler/vllm.cpp/issues/412)):

| Key | Value taken when absent |
|---|---|
| `qk_scale_factor` | 43.784, which is 3.87 at `head_dim` 128 |
| `sliding_window` | 2048, not "no window at all" |
| `output_multiplier` | 0.196... |
| `final_logit_softcapping` | 20.0 |
| `rms_norm_eps` | 1e-5 |
| `post_norm_eps` | 1e-8 |

The released 30B `config.json` carries all six, so the safetensors arm is
unaffected. The released GGUF and the DFlash drafter's `config.json` each omit
some, and both used to run a quietly different model. The released file's 32
metadata keys include no post-norm epsilon, so both sandwich post-norms used to
run at `attention.layer_norm_rms_epsilon` (1e-5) where the architecture says
1e-8, a factor of 1000.

Correcting this changes GGUF activations, though a same-binary A/B on the
released k-quant produced **token-identical** greedy output on both prompts on
record. Only an explicit `null` still disables the window or the soft-cap. A
converter that emits `muse-glimmer.attention.post_norm_rms_epsilon` or
`muse-glimmer.attention.scale` is honored over the default.

## What has actually been checked

- The text tower ran on real tensors from the released 30B checkpoint at
  **reduced depth, 4 of its 52 layers.** Its **5 prefill argmax positions** are
  identical to a standalone torch transcription of the upstream source and to
  Hugging Face's own `muse_glimmer` implementation. The full-depth 52-layer
  bf16 safetensors arm has generated once, 4 ungated tokens (`" Paris. It is"`,
  2026-08-11); no bf16 token gate exists. The full-depth GGUF Q4_K_M arm is
  token-gated (see the llama.cpp token gate above).
- Those are argmax positions from a single prefill, not generated tokens.
  **Multi-step decode on the bf16 safetensors arm is ungated**; on the GGUF
  arm it is gated for 32 tokens (prompts are far shorter than the sliding
  window, so the window across steps is still untested).
- Even at reduced depth, this is agreement with independent transcriptions of the
  same upstream source, not agreement with the model's own runtime. The vLLM
  parity pin `a7c23ac96d` registers `MuseGlimmerForConditionalGeneration`, but
  no bf16 token gate against it has run (it needs a GPU lease; owed).
- The perception encoder has **one reference run**, on the released tensors
  (2026-09-30, `test_muse_glimmer_vision_real`, env-gated): a 588x644 image,
  483 soft-token rows, against a torch transcription of the pinned formulas.
  f32: `ln_pre` 1.22e-6, tower 7.87e-5, soft tokens 4.07e-5 relative max error.
  bf16 against the f32 truth: worst row cosine 0.967, mean 0.99875, where the
  reference's own bf16 arm reaches 0.970 / 0.99856. The image processor is not
  ported (the tower is fed the reference's pixels), and no image-to-text result
  exists. The encoder normalizes merged multimodal embeddings again as of #405.
  Its config key is `null` in the released checkpoint, which the pinned config
  class reads as on. It had been read as off, so image and video prompts before
  that fix skipped a normalization step.
- **Nothing has run end to end through the server**, and **no speed number exists
  for this model in any weight format** against vLLM: the parity pin now
  registers the model, but its speed on this fleet is unmeasured, so no
  denominator is quoted.
- The ATEM reasoning and tool parsers are ported and unit-gated. At the server's
  default `skip_special_tokens: true` the framing tokens they key on
  (`<|start|>`, `<|message|>`, `<|eom|>`, `<|eot|>`) are stripped before the
  parser sees the text, so channel scoping is an **open gap at server defaults**.
  See [`docs/FEATURES.md`](../FEATURES.md) and
  [the spec](../../.agents/specs/muse-glimmer.md) section 6.7.

## Run the gate against a real checkpoint

Set `VLLM_MUSE_GGUF=<file>`, or `VLLM_MUSE_GGUF_LOAD=<file>` for the full
materialization, to run `test_muse_glimmer_gguf` against a real checkpoint.
Without them the gate runs off committed header-only manifests.
