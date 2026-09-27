ID: ISSUE-GH-2279
Title: **`FromGguf` never reads `tokenizer.ggml.add_bos_token`, so a GGUF that asks for a leading BOS silently gets none.** Found while adding the `glm4` pre name for [#2277](https://github.com/mudler/vllm.cpp/issues/2277) and deliberately not fixed in that flow: #2277's scope is one pre name, this is a property of every GGUF tokenizer this tree loads. llama.cpp reads the key at `b10451:src/llama-vocab.cpp:2585-2586`, and `add_bos` is the ONLY thing that decides the prepend (`:3382-3384`, `if (add_special && add_bos)`); `tok::Tokenizer::FromGguf` reads `tokenizer.ggml.bos_token_id` and stops there, leaving `template_bos_` at -1 so `EncodeWithSpecialTokens` reduces to `Encode` for every GGUF. Nothing is red today because no artifact this tree gates on states the key -- the staged `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL carries 72 KV entries and it is not among them, parsed 2026-08-29 from shard 1's own KV block, so llama.cpp's `add_bos` stays at its `:1815` default `false` and our silence is the right answer there. It is already live in the other direction on the `llama-bpe` family, whose arm at `:2157-2159` sets `add_bos = true` where the `glm4` arm at `:2256-2259` sets nothing, masked only because that path has never been token-gated against llama.cpp with `add_special = true`. No gate can see this class of defect: a prompt short by exactly one leading token still decodes to fluent text, still has a valid shape, still loads and still generates, and a token gate built from our own tokenizer compares us against us. Scope: read `add_bos_token` (and `add_eos_token`, the same upstream block) defaulting to llama.cpp's `false`; decide what represents it, since `template_bos_` has the right meaning and the wrong provenance comment; a case per arm proving exactly one BOS when true, none when false or absent, both round-tripping; and enumerate which committed fixtures and staged artifacts declare the key so the blast radius is measured rather than assumed. Recorded as O21 in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md)
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 2279
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:883`

### Frozen archive evidence

> | [#2279](https://github.com/mudler/vllm.cpp/issues/2279) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **`FromGguf` never reads `tokenizer.ggml.add_bos_token`, so a GGUF that asks for a leading BOS silently gets none.** Found while adding the `glm4` pre name for [#2277](https://github.com/mudler/vllm.cpp/issues/2277) and deliberately not fixed in that flow: #2277's scope is one pre name, this is a property of every GGUF tokenizer this tree loads. llama.cpp reads the key at `b10451:src/llama-vocab.cpp:2585-2586`, and `add_bos` is the ONLY thing that decides the prepend (`:3382-3384`, `if (add_special && add_bos)`); `tok::Tokenizer::FromGguf` reads `tokenizer.ggml.bos_token_id` and stops there, leaving `template_bos_` at -1 so `EncodeWithSpecialTokens` reduces to `Encode` for every GGUF. Nothing is red today because no artifact this tree gates on states the key -- the staged `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL carries 72 KV entries and it is not among them, parsed 2026-08-29 from shard 1's own KV block, so llama.cpp's `add_bos` stays at its `:1815` default `false` and our silence is the right answer there. It is already live in the other direction on the `llama-bpe` family, whose arm at `:2157-2159` sets `add_bos = true` where the `glm4` arm at `:2256-2259` sets nothing, masked only because that path has never been token-gated against llama.cpp with `add_special = true`. No gate can see this class of defect: a prompt short by exactly one leading token still decodes to fluent text, still has a valid shape, still loads and still generates, and a token gate built from our own tokenizer compares us against us. Scope: read `add_bos_token` (and `add_eos_token`, the same upstream block) defaulting to llama.cpp's `false`; decide what represents it, since `template_bos_` has the right meaning and the wrong provenance comment; a case per arm proving exactly one BOS when true, none when false or absent, both round-tripping; and enumerate which committed fixtures and staged artifacts declare the key so the blast radius is measured rather than assumed. Recorded as O21 in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md) | bug |

## Resolution

-
