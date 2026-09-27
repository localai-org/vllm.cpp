ID: ISSUE-GH-1198
Title: Three specs assert `model_loader.cpp` behaviour the loader no longer has, found while verifying line citations for conversion under [#1143](https://github.com/mudler/vllm.cpp/issues/1143): `safetensors-windowed-load.md:63,108` quotes a `shards.clear()` that does not exist (the shard vector is a `shared_ptr` released by the deferred-expert closure, `model_loader.cpp:1636-1643`), `gguf-dflash-draft.md:17-18` calls `LoadDflashDraft` "still typed on `std::vector<SafetensorsFile>`" when it takes a `SharedHeadSource`, its `A5`/`B2` rows plan around a GGUF refusal the loader says at `:906-910` is GONE, and `model-factory-registry.md:91` cites `IsDenseArch` which survives only in a comment saying the registry superseded it. Filed rather than repaired because a citation sweep can see the claim is false and not what the true statement is — that is the owning row's judgement. Owned under `## Owed` in [`citation-anchor-freshness.md`](../../specs/citation-anchor-freshness.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1198
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:393`

### Frozen archive evidence

> | [#1198](https://github.com/mudler/vllm.cpp/issues/1198) | — | Three specs assert `model_loader.cpp` behaviour the loader no longer has, found while verifying line citations for conversion under [#1143](https://github.com/mudler/vllm.cpp/issues/1143): `safetensors-windowed-load.md:63,108` quotes a `shards.clear()` that does not exist (the shard vector is a `shared_ptr` released by the deferred-expert closure, `model_loader.cpp:1636-1643`), `gguf-dflash-draft.md:17-18` calls `LoadDflashDraft` "still typed on `std::vector<SafetensorsFile>`" when it takes a `SharedHeadSource`, its `A5`/`B2` rows plan around a GGUF refusal the loader says at `:906-910` is GONE, and `model-factory-registry.md:91` cites `IsDenseArch` which survives only in a comment saying the registry superseded it. Filed rather than repaired because a citation sweep can see the claim is false and not what the true statement is — that is the owning row's judgement. Owned under `## Owed` in [`citation-anchor-freshness.md`](../../specs/citation-anchor-freshness.md) | bug |

## Resolution

-
