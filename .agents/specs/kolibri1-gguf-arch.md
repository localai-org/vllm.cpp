# kolibri1 GGUF arch — the published Hob-forge container loads

Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
Issue: ISSUE-LOCAL-01M5HJRRHBYMRX4G4JXXKCBFJS
Decision record: commit 39dadc637 (ROUTE A) on row/KOLIBRI1-GGUF-ARCH

## Scope

The `kolibri1` GGUF arm on the CPU path:

- the arch row in the top-level GGUF dispatch table
  (`kGgufArchArms`, `src/vllm/entrypoints/model_loader.cpp:1278`);
- the weights TU that loads the published Hob-forge container into the
  EXISTING `Kolibri1Weights` the fp8 safetensors arm already fills
  (`src/vllm/model_executor/models/kolibri1_weights.h`);
- the k-quant dequant branch in the load path (Q5_0 / Q6_K / F32, all
  already implemented in `src/vllm/model_executor/model_loader/gguf_dequant.cpp`);
- the `kolibri1` pre-tokenizer mapping in `Tokenizer::FromGguf`
  (`src/vllm/tokenizer/tokenizer.cpp:984`).

Out of scope: the Tenstorrent GGUF arm, keep-quant residency for this
architecture, the FLINT comparison container (`/mnt/models/KOLIBRI1-GGUF-HANDOFF.md`),
and any CUDA device arm.

## Upstream anchors

- The container is NOT llama.cpp-conformant and llama.cpp is NOT the floor:
  `expert_gating_func = 5` aborts llama.cpp's `build_moe_ffn`
  (src/llama-graph.cpp:2218 at b11765). ROUTE A (recorded in the issue and in
  commit 39dadc637) gates the file AS PUBLISHED; the file's bytes are not
  modified and its sha256 stays the published one
  (Q2_K de2fac52..., Q4_K_M c2ac1301..., repo 40eaa5cf).
- The working reference implementation is mistral.rs (the issue's floor for
  this container): the pre-type `kolibri1` maps to its KOLIBRI1_REGEX
  (mistralrs-core/src/gguf/gguf_tokenizer.rs:364), byte-identical to this
  tree's `kClassicQwen2N1Regex` (src/vllm/tokenizer/tokenizer.cpp:37).
- llama.cpp maps Kolibri-1's tokenizer to qwen2 by sha256 (llama.cpp PR
  #30673); the tensor-name reference is llama.cpp PR #30928 (`kolibri1`:
  token_embd / output / output_norm, blk.N.attn_{q,k,v,output},
  attn_{q,k}_norm, attn_norm, post_attention_norm, post_ffn_norm,
  ffn_gate_inp, exp_probs_b, ffn_{gate,up,down}_exps,
  ffn_{gate,up,down}_shexp).
- Routing is FIXED by the model, not read from the file: sigmoid, raw
  unscaled top-k weights (kolibri1.py:126-142); mistral.rs's `build_kolibri1`
  never reads `expert_gating_func` either. Our loader does NOT consult the
  key.

## Design

ONE row, ONE weights TU, ONE pre-type mapping.

1. **Dispatch.** `{"kolibri1", &vllm::Kolibri1HfConfigFromGguf}` in
   `kGgufArchArms`. The builder synthesizes an HF-shaped `HfConfig` handed to
   the SAME `ParseKolibri1Params` a config.json descends through
   (`src/vllm/model_executor/models/kolibri1_registry.cpp:62`), so both
   sources meet one validator. `LoadKolibri1ForCausalLM` replaces its
   "GGUF is not supported" refusal with the `kGguf` branch calling
   `LoadKolibri1FromGguf` (the qwen4_exp_registry.cpp:96 pattern).

2. **Config mapping** (file KV -> HF field). `attention.key_length` ->
   `head_dim`; `attention.sliding_window` -> `sliding_window` with
   `use_sliding_window = true`; `attention.sliding_window_pattern[i]` ->
   `layer_types[i]` (`true` is `sliding_attention`, corroborated field-for-
   field against the safetensors config.json's `layer_types` on the real
   checkpoint); `rope.freq_base` -> `rope_theta`; `attention.layer_norm_rms_epsilon`
   -> `rms_norm_eps`; `expert_count`/`expert_used_count`/
   `expert_feed_forward_length`/`expert_shared_feed_forward_length` -> the MoE
   fields; `expert_weights_norm` -> `norm_topk_prob` (both false on the
   published file — the same no-renormalisation contract the parser already
   requires); `vocab_size` from `token_embd.weight`'s leading dim.
   `expert_gating_func` is NOT read, per the recorded decision.

3. **Weights TU** (`kolibri1_gguf_weights.{h,cpp}`). Loads into the existing
   bf16-owned `Kolibri1Weights`: every projection becomes the `bf16` arm of
   `Kolibri1Projection`, so the production forward's `LinearBT`/`LinearBTRaw`
   dispatch (kolibri1_forward.cpp:285,316) is unchanged and reaches the load
   with no new seam. GGUF stores ne reversed; our reader already hands back
   torch row-major `[out, in]` (gguf_reader.cpp:515-519), which is exactly
   the MatmulBT orientation `LinearBTRaw` consumes — so every 2-D weight is
   a verbatim dequant + copy, no transpose. The expert towers are stacked
   `[E, out, in]`; expert `e` is rows `[e*out, (e+1)*out)` of the flattened
   dequant. The router gate `ffn_gate_inp` arrives `[out=384, in=2560]`
   directly. `exp_probs_b` is F32 and stays F32 (the safetensors loader also
   widens to f32); every RMS-norm gamma is F32 on disk and converts to the
   bf16 the forward's `ResidentWeight` scratch consumes.

4. **k-quant branch.** The load expands through
   `DequantGgufRowToBf16` (`gguf_dequant.cpp`), which already carries Q5_0
   and Q6_K. Q2_K is a Q5_0/Q6_K/F32 file (per-tensor types read from the
   file itself), so no new block decoder is owed.

5. **Pre-tokenizer.** `pre == "kolibri1"` -> `SplitPattern::kQwen2Classic`
   beside the existing `qwen2` arm, anchored on mistral.rs's KOLIBRI1_REGEX
   being byte-identical to `kClassicQwen2N1Regex` (already mapped onto
   kQwen2Classic at tokenizer.cpp:512) and on llama.cpp PR #30673's qwen2
   sha256 match. This is an EXACT mapping, not an approximation: the
   regexes are byte-equal.

## Risks

- **Memory.** The Q2_K expands to ~157 GB of bf16 (78.4 B params). The gate
  host carries 255 GB (219 available), so the load fits with the dequant
  temporaries (~1 GB per expert tower) freed per tensor. A host with less
  RAM refuses at allocation, not silently.
- **No keep-quant.** The kolibri1 forward has no block-aware kernel, so
  expanding is the only arm that is CORRECT, not merely convenient. Keep-quant
  for this architecture is owed separately (see ## Owed) and must land in the
  forward, not in a loader that keeps blocks nothing can decode.
- **Interpretation risk** (`expert_gating_func = 5`, `expert_weights_norm =
  false` against the config's `norm_topk_prob: true`): carried by ROUTE A and
  adjudicated by the token gate below, not by this spec.
- **near-tie band.** The gate reuses the W3 band (2.5 nats,
  tests/vllm/models/test_kolibri1_w3.cpp:56), itself derived from the
  measured 2.19 bf16 envelope. The k-quant arm adds dequant rounding on top
  of bf16; if the chain diverges hard, the band is re-measured against the
  primary oracle, never widened ad hoc.

## Tests

- `tests/vllm/models/test_kolibri1_gguf.cpp`:
  - RED FIRST: `LoadKolibri1FromGguf` (or, before the loader exists, the
    engine's GGUF resolve) refuses the published Q2_K today as an
    unrecognized architecture. The test captures the refusal, then asserts
    the load succeeds after the row lands.
  - THE TOKEN GATE: the same 8 golden prompts
    (`tests/vllm/models/kolibri1_goldens.json`), the same teacher-forced
    fingerprint and greedy-chain adjudication as test_kolibri1_w3.cpp, over
    the published Q2_K. Pass condition: no hard flip (every divergence inside
    the recorded band). Q4_K_M is the same code path (same encodings) and is
    not separately gated here.

## Gates

- `scripts/agent-preflight.sh` and `--staged`.
- `check-agent-record.py`, `check-env-doc.py`, `check-site.py`,
  `check-oracle-pins.py`, `check-commit-style.py --range forge-main..HEAD`,
  `check-commit-trailers.py --range forge-main..HEAD --filled`.
- `check-windows-portability.py` fails on THIS host on a pristine tree
  (dotnet/fake-tool) — a host property, reported as such.
- Build on CPUs 0-15 only, `nice -n 19`; no page-cache-dropping reads of the
  model files; every long run bounded.

## Owed

- Keep-quant (block-resident) expert residency for the kolibri1 forward —
  the expand-to-bf16 arm holds ~157 GB where the file is 27-45 GB. Owed in
  the forward (a block-aware LinearBT arm), not in the loader.
- The Q4_K_M container is not separately gated (same encodings as Q2_K); a
  recorded run against it is owed when a GPU/long-run window is available.
- CUDA/other-device GGUF arms: unowned, refused by name by the existing
  device resolution.
- The FLINT comparison quantization (`/mnt/models/KOLIBRI1-GGUF-HANDOFF.md`)
  is a separate developer-owned artifact; when it lands it needs no loader
  change beyond its own row if its arch key differs.

## Evidence

- Published files: /mnt/models/Hob-forge-Kolibri-1-GGUF/{Kolibri-1-Q2_K,Q4_K_M}.gguf,
  sha256 sidecars beside them; metadata dumped with /tmp/gguf_meta.py
  (2026-10-23): arch `kolibri1`, 50 blocks, hidden 2560, 48/4 heads,
  head_dim 128, sliding 513, 384/6 experts, ff 512/512, shared count 1,
  expert_weights_norm false, expert_gating_func 5 (unread), pre `kolibri1`.
- Goldens: tests/vllm/models/kolibri1_goldens.json (8 prompts, greedy, 32
  tokens, torch-f32 transcription; band 2.5, test_kolibri1_w3.cpp).

## Outcome (2026-10-23, first implementation wave)

Landed: the dispatch row, the weights TU, the dequant branch, the pre-type
mapping, and the tests. RED FIRST captured: with the dispatch row reverted,
`test_model_loader_gguf`'s "a kolibri1 GGUF reaches ITS OWN builder" fails —
the file is refused as "is not supported by this build"; with the row, the
suite passes 15/15 (66 assertions). The real-file config gate passes
(63 assertions: geometry, the 4:1 sliding pattern off the file's own
sliding_window_pattern, norm_topk_prob=false off expert_weights_norm).

THE TOKEN GATE AGAINST THE BF16 GOLDENS **FAILS, AND IS REPORTED AS A
FAILURE, NOT WAIVED**: ARGMAX CHAIN 4/12 match, 4 flips near-tie, **4 hard
flips**; worst teacher-forced topk diff 4.96 against the 2.5-nat band. The
band was measured for bf16 rounding (2.19 envelope, test_kolibri1_w3.cpp);
a 2.56-bpw container's dequant error is a different, much larger envelope,
and no band extension was applied. The fluency probe (VT_KOLIBRI1_GGUF_GEN=1)
shows fluent, prompt-appropriate, degenerate-looping text of exactly the
shape the bf16 goldens themselves show in this greedy short-prompt regime
(the bf16 golden for "Ein gutes Buch liest man am besten" loops the same
phrase), so the load is structurally right and the divergence is the
quantization.

ADJUDICATOR STATUS — the gate has no runnable floor for this file:
- llama.cpp at b11765 aborts on expert_gating_func=5 (the ROUTE A premise).
- mistral.rs at the peer's tree REFUSES the file: build_kolibri1
  (mistralrs-core/src/gguf/normal_config.rs:1654) goes through
  StandardFields::read (normal_config.rs:931), which requires
  `kolibri1.feed_forward_length`, absent from BOTH published containers.
  The recorded "mistral.rs loads and runs it" (issue correction, 2026-10-23)
  is not reproducible on the published Q2_K/Q4_K_M via the standalone GGUF
  path; the recorded 8.0 tok/s benchmark was a UD-Q4_K_XL container this box
  does not have.
NEEDS_DECISION for the gate's adjudicator: (a) a derived copy of the
published file with `feed_forward_length` added (a NEW container, published
bytes untouched) so mistral.rs runs it and the parity gate is our-chain vs
mistral.rs-chain on identical weights; (b) a MEASURED quantization envelope
band (re-run the fingerprint methodology against a bf16-dequantized
reference of the SAME container) replacing the bf16 band for this arm; or
(c) Route B — the handoff's llama.cpp-conformant container, where the
existing band applies unchanged.
