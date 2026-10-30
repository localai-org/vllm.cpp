# SPEC — `KOLIBRI1-GGUF-ARM` — the kolibri1 GGUF arch loads and serves on CPU; the mudler APEX tiers are the shipped artifacts

- Owning row: `KOLIBRI1-GGUF-ARM`
- Stable matrix row: `QUANT-KOLIBRI1-GGUF-ARM` (quantization-matrix, the
  checker's row-ID namespace requires the `QUANT-` prefix; this spec is that
  row's spec)
- Kind: feature (model port coverage: the quantized arm the protocol requires)
- Issue: created in the same flow (row-scoped local issue)

## The artifact

`mudler/Kolibri-1-APEX-GGUF` @ `55a4781439f1e8abb7e46c795e0681d6b2b7f4d9` —
nine tiers, 22.17 GB (I-Micro) to 56.21 GB (Balanced), standard ggml types
only: Q2_K token_embd, Q5_K output + shexp, Q4_K attention, Q3_K/Q4_K +
IQ1_M + IQ2_XS routed experts, F32 norms. Header verified by range-request
parse (2026-10-30): `general.architecture = kolibri1`, 903 tensors, 50
blocks, 384 experts / 6 used, hidden 2560, 48/4 heads, sliding window 513,
`expert_shared_count = 1`, `expert_weights_norm = 0`. I-Micro downloading to
`/mnt/models/kolibri1-apex-gguf/` (sha256
`3edfaf09aac60d60cde56744fe1e0a7e5362c5549d5eb12d97ec4a9c83c90e0e`); fetch
log /tmp/kq-apex-fetch.log.

## The blockers (both verified in-tree; this row removes them)

1. **No arch row**: the GGUF loader dispatch table
   (`src/vllm/model_executor/models/model_loader.cpp:1221`, `GgufArchArm`)
   has no `kolibri1` entry — a Kolibri-1 GGUF is refused as unrecognized.
2. **Quant acceptance**: `kolibri1_weights.cpp` `CheckQuantConfig`
   (`:64-80`) refuses any checkpoint without the fp8-block [128,128]
   dynamic `quantization_config`. The GGUF arm arrives as k-quants/IQ and
   must be admitted for the GGUF arm while the fp8 requirement stays for
   the safetensors arm.

## Design intent

Follow the house GGUF-weights-TU shape (`qwen4_exp_gguf_weights.cpp` /
`glm5_next_weights.cpp`): an arch row + a `kolibri1_gguf_weights.cpp` TU
that maps the 903 tensors (token_embd, output, output_norm, attn q/k/v/out,
attn_norm, post_ffw_norm, ffn_gate/up/down_exps, ffn_gate/up/down_shexp,
plus the router/probs tensors the header carries) into the existing CPU
kolibri1 forward's weight structs, dequantizing at load through
`gguf_dequant.cpp` (all types already known). The forward itself is
unchanged: the CPU kolibri1 row is shipped and gated; the GGUF arm feeds it
dequantized weights. Red-first: loading the I-Micro file at HEAD → refused
as unrecognized arch (capture it).

## Oracles

- **Primary (behavior reference)**: the shipped CPU fp8 arm — token
  cross-check GGUF-vs-fp8 on identical prompts, in-band (the
  `test_kolibri1` contract shape). Different kernels/compute classes, so
  the rel_rms/band contract governs, not bit-exactness.
- **Floor (throughput)**: llama.cpp supports kolibri1 since PR #30673
  (merged 2026-10-17) with #30865 (router precision removal, 2026-10-20).
  The `llama-cpp` pin (b10451) predates both → a `llama-cpp-kolibri1`
  secondary record (the qwen4exp/glm5next shape) is owed IF a floor
  comparison is wanted; the ARM does not block on it — record the
  pin-advance as a named follow-up.

## Non-goals

- No TT-path change (the P150 takes k-quants via a different, separately
  owed row).
- No new dequant kernels (all types known).
- No GPU/CUDA arm.

## Red-first

At HEAD, `LoadKolibri1`-path on the I-Micro file → unrecognized-arch
refusal. Green-after: the file loads, serves greedy tokens in-band vs the
fp8 arm, and a throughput number at 16/32 threads is recorded next to the
floor when the llama-cpp pin advances (or alone, labeled, if not).

## Gates

check-agent-record, style, trailers, env-doc (if any new env), preflight
(named host dotnet failure); CI green; the GGUF-arm serving number +
cross-check recorded under docs/bench-evidence/.

## Stop conditions

- If the tensor set cannot map onto the CPU forward's structs without
  forward-logic changes (e.g. a tensor the header carries that the fp8
  checkpoint lacks, like `post_ffw_norm` vs the fp8 arm's norm placement),
  STOP and report — that is a graph-shape discovery, adjudicated before
  proceeding.
- If the in-band cross-check fails beyond the fp8-vs-GGUF class difference,
  STOP: numerics first.

## Git integration

One pull request, `row/KOLIBRI1-GGUF-ARM`, spec committed first.

## Scope (structured record, added by the implementer 2026-10-30)

Gate the mudler APEX I-Micro tier through the CPU kolibri1 GGUF arm; record
the fp8-oracle cross-check, 16/32-thread throughput, evidence, and the
weight pin. Out of scope: TT path, new dequant kernels, GPU arms.

## Upstream chain

`mudler/Kolibri-1-APEX-GGUF` @ `55a4781439f1e8abb7e46c795e0681d6b2b7f4d9`
(quantized from `Aleph-Alpha/Kolibri-1-BF16`); primary oracle the fp8
`Aleph-Alpha/Kolibri-1` safetensors arm via `kolibri1_goldens.json`.

## Our baseline

Both named blockers were already closed by commits `71ee347e9` +
`0350841b7` (2026-10-23): the arch row at
`src/vllm/entrypoints/model_loader.cpp:1277` and the
`kolibri1_gguf_weights.cpp` TU, which keeps the fp8 checks scoped to the
safetensors loader by construction.

## Port map

No product code. `tests/CMakeLists.txt` gained a `KOLIBRI1_GGUF_PATH` cache
knob so the committed gate binary can point at any kolibri1 GGUF container
(default unchanged: the Hob-forge Q2_K).

## Tests to port

None — the harness exists (`tests/vllm/models/test_kolibri1_gguf.cpp`, the
W3 contract shape); this row points it at a new artifact.

## Dependencies

The I-Micro artifact (downloaded + sha256-verified 2026-10-30); the fp8
oracle checkpoint mounted at `/mnt/models/Aleph-Alpha/Kolibri-1`.

## Work breakdown

1. Record reconciliation + red-first disposition (done: red state is
   historical, pre-`39dadc637`).
2. Artifact fetch + header census (done).
3. Load + structural accounting (done: 903/903 mapped).
4. Serving cross-check — STOP HIT (worst topk diff 3.245 > 2.5 band, 2 hard
   flips; fluency clean; evidence
   `docs/bench-evidence/kolibri1-gguf-arm-20261030/`).
5. Throughput 16/32t + public-doc weight pin — WITHHELD pending the
   developer's numerics adjudication.

## Risks/decisions

The band-vs-quantized-arm adjudication is deliberately left open (the
kolibri1-gguf-arch row left the identical question open); this row does not
widen a band to go green.
