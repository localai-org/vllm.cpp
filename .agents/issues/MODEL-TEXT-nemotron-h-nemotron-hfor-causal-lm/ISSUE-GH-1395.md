ID: ISSUE-GH-1395
Title: A2-B: NemotronH refuses every step carrying more than one request (`nemotron_h_registry.cpp:159-167`), four specs name A2-B as the owner of that clause, and no spec for A2-B existed. Because vLLM's production configuration batches, an engine capped at `num_reqs <= 1` cannot be the numerator of any throughput ratio against it, so every Nemotron speed number is bounded until the clause lifts. Spec [`nemotron-h-a2b-batched-decode.md`](../../specs/nemotron-h-a2b-batched-decode.md) writes the unit up against the pinned oracle at `555967922` and finds the work is not what the refusal's wording suggests: A2-P's §4.1 (`nemotron-h-a2p-paged-forward.md:348-354`) already landed the per-request state indexing (`GatherNemotronHState` / `ScatterNemotronHState` / the `idx`-`init` build are all `R`-general), and `runner.cpp:1257` already reorders decode-first on upstream's own predicates. What is genuinely missing is a MIRRORING gap the refusal has been hiding: `mamba_attn.py:467` passes `treat_short_extends_as_decodes=False` and never passes `require_uniform` (16 call sites at the pin, 8 pass `require_uniform`, `mamba_attn.py` is not one), while our shared `SplitDecodesAndPrefills` (`gdn_attn.cpp:12-13`) implements the `True` mode that is correct for `gdn_attn.py:213` and wrong for this architecture — unobservable at one request, and fluent wrong tokens at two. The population that discriminates the two modes is NOT a short extend, on which the promotion at `mamba_attn.py:450-453` always fires, but a still-prefilling row with `query_len == 1` and no prior state (`seq_len == 1`). Umbrella [#810](https://github.com/mudler/vllm.cpp/issues/810)
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: feature
GitHub: 1395
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:468`

### Frozen archive evidence

> | [#1395](https://github.com/mudler/vllm.cpp/issues/1395) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | A2-B: NemotronH refuses every step carrying more than one request (`nemotron_h_registry.cpp:159-167`), four specs name A2-B as the owner of that clause, and no spec for A2-B existed. Because vLLM's production configuration batches, an engine capped at `num_reqs <= 1` cannot be the numerator of any throughput ratio against it, so every Nemotron speed number is bounded until the clause lifts. Spec [`nemotron-h-a2b-batched-decode.md`](../../specs/nemotron-h-a2b-batched-decode.md) writes the unit up against the pinned oracle at `555967922` and finds the work is not what the refusal's wording suggests: A2-P's §4.1 (`nemotron-h-a2p-paged-forward.md:348-354`) already landed the per-request state indexing (`GatherNemotronHState` / `ScatterNemotronHState` / the `idx`-`init` build are all `R`-general), and `runner.cpp:1257` already reorders decode-first on upstream's own predicates. What is genuinely missing is a MIRRORING gap the refusal has been hiding: `mamba_attn.py:467` passes `treat_short_extends_as_decodes=False` and never passes `require_uniform` (16 call sites at the pin, 8 pass `require_uniform`, `mamba_attn.py` is not one), while our shared `SplitDecodesAndPrefills` (`gdn_attn.cpp:12-13`) implements the `True` mode that is correct for `gdn_attn.py:213` and wrong for this architecture — unobservable at one request, and fluent wrong tokens at two. The population that discriminates the two modes is NOT a short extend, on which the promotion at `mamba_attn.py:450-453` always fires, but a still-prefilling row with `query_len == 1` and no prior state (`seq_len == 1`). Umbrella [#810](https://github.com/mudler/vllm.cpp/issues/810) | feature |

## Resolution

-
