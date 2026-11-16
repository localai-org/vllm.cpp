# Mach-2 native packed (`mach2-native`)

A **secondary oracle** under "when vLLM has no implementation". vLLM implements
no trellis codec, no additive codeword and no wave-scaled tile, so there is no
primary denominator for this format.

| Field | Value |
|---|---|
| Oracle | `SyzygyResearch/Mach-2-Additive-Medium` (native packed weights) |
| Registry id | `mach2-native` |
| Format tag | `fn_packed_v2`, asserted by every expert shard's `format` metadata |
| Format definition | `decode.py` in that repository, served from `main` |
| Base checkpoint | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Text weights | 26.7 GB at 1.70 bits per weight, 125.7B parameters |
| Alongside | n-gram embedding table at bf16, 102.4 GB; vision 0.9 GB at bf16 |
| Read on | 2026-11-16 |
| `gateable` | **no** |

## Why `gateable = no`

An oracle is gateable once it demonstrably builds and runs the model. Nothing has
been run here: no weight was downloaded, no decoder was executed, and no GPU was
leased. What is established is the **format**, read from the release decoder's
own source. [`.agents/specs/quant-mach2-additive.md`](../specs/quant-mach2-additive.md)
records the analysis.

Two facts must be measured before this flips to `yes`:

1. That the published `decode.py` runs (it must `py_compile` and decode a real
   shard). The text fetched on 2026-11-16 loses leading whitespace in places,
   which would be an `IndentationError`; whether that is the file or the fetch is
   **unverified**.
2. That a leased host can hold the artifact: ~26.7 GB of weights on a 32 GB GPU
   plus ~105 GB of n-gram table in host RAM.

## What it is not

The GGUF sibling `SyzygyResearch/Mach-2-Additive-Medium-GGUF` (130.1 GB) is **not**
this oracle. It requires `SyzygyResearch/llama.cpp-mach1`, whose `docs/mach1.md`
documents payload v3 and eight `MACH1_*` ops for **Mach-1**, and names no Mach-2
payload version. It is a separate potential oracle with its own pin, owed by the
GGUF arm of the row rather than by this file.

## Determinism

The release decoder is pure numpy over integer codes: the state walk, LUT
expansion, `wave_gamma` gather, `rc_grid`, int5 unpacking and grouped int4 are
exact integer/dtype operations, and the only rounding is the deliberate
`bf16_round` in the spine path. A port that reproduces those operations in the
stated order is byte-exact, so this oracle supports a **byte-exact** gate rather
than a tolerance gate — which is why the spec refuses to weaken it.

The one thing it cannot supply is a token stream: the model card states the model
loops without temperature 1.0 / top_p 0.95 / top_k 20, so a greedy token-exact
gate may not exist for it. The decode gate comes first for that reason.
