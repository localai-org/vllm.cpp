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

## Pin

```oracle-pin
id = mach2-native
role = secondary
upstream = https://github.com/SyzygyResearch/Mach-2-Additive-Medium
scope = the Mach-2 additive codec only: a QTIP bitshift trellis whose codewords are D4 lattice integers applied with addition, the fn_packed_v2 packing that stores it, the per-tile wave scales and the radix-12/20 Hadamard rotations, which neither vLLM nor vLLM-Omni implements in any part
pin = UNPINNED
pin_label = main, 2026-11-16 (decode.py sha256 e27dbbe5)
pinned_on = 2026-11-16
gateable = no
evidence = #3573
```

`pin` stays `UNPINNED` and that is a measurement, not a shorthand: the repository
is not reachable from an authenticated GitHub token on 2026-11-18, so no commit of
`main` can be resolved and this record will not invent one. What identifies the
format instead is the file's bytes: `decode.py` read on 2026-11-16 has 17,766 of
them and sha256 `e27dbbe500250364506f8b7e8afb46b7c86971c782c7e0824e49a851a1158b1e`,
recorded in the spec. A pin by content is weaker than a pin by revision — it
cannot be advanced, diffed, or followed — and it is named here rather than dressed
up as a commit.

## Why `gateable = no`

An oracle is gateable once it demonstrably builds and runs the model. The model
has not been run: no weight was downloaded and no GPU was leased. What is
established is the **format**, read from the release decoder's own source, and
that decoder's agreement with our fixtures.
[`.agents/specs/quant-mach2-additive.md`](../specs/quant-mach2-additive.md)
records both.

**The decoder's own runnability is retired, and this record said otherwise.** The
earlier version of this section carried the mangled indentation of a 2026-11-16
fetch as an open risk. The file downloads clean at 17,766 bytes with LF endings
and intact indentation, `python3 -m py_compile decode.py` exits 0, and U1 and U2
then ran that unmodified file over our fixtures — 51,184/51,200 values bit-identical
at 160x320 (radix 20) and 4,601/4,608 at 96x48 (radix 12), residual about one ulp.
The mangling was the fetch.

Two facts about the ARTIFACT are still owed before this flips to `yes`, and
[#3573](https://github.com/localai-org/vllm.cpp/issues/3573) owes both:

1. That a **real published shard** decodes. Every fixture above is one we
   generated; no file from the 26.7 GB text-weights release has been downloaded,
   so the `format = fn_packed_v2` metadata every expert shard is said to carry
   has never been read from the artifact that carries it.
2. That a leased host can hold the artifact: ~26.7 GB of weights on a 32 GB GPU
   plus ~102.4 GB of bf16 n-gram table in host RAM.

Until both land, the decode gate rests on a format read from source plus a
differential measured on synthetic code streams — an oracle that has never seen
the weights it is supposed to gate.

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
