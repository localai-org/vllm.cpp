ID: ISSUE-GH-1666
Title: **The O26C `attn_groups` census records 30 GDN layers; the run resolved 48, and 30 is the TEST STAND-IN's shape.** Four places -- `.agents/specs/dflash2-spec-decode.md` `## Owed` O30, the `BACKEND_GROUP_PROBES` comment in `tools/bench/dflash2_speed_harness.py`, `resolve_attention_backend_groups` in `tools/bench/dflash2_oracle_capture.py`, and `test_ONE_SCALAR_UNDER_DESCRIBES_this_model_so_the_MAP_is_recorded_too` -- recorded the 2026-08-22 leased walk as `GDNAttentionBackend` over **30** `linear_attn` layers. Re-derived from the run's own `c-probe-result.json`, under `candidate_walks[...worker.model_runner.attn_groups]`, by parsing every `AttentionGroup(backend=..., layer_names=[...])` pair and asserting the match count equals the 15 occurrences of `AttentionGroup(backend=`: **48 GDN layers in 10 groups** (eight of five, two of four), 16 `self_attn.attn` in 4 groups at every 4th index 3-63, and 5 draft layers (`model.layers.64-68`) in 1. The 16 and the 5 were right; only the 30 was wrong, and 30 is `_stand_in_attn_groups()`'s `range(30)` -- a fixture's arbitrary shape generalised into a measurement claim, on the row whose whole history is that failure class. The tree already disagreed with itself: `.agents/specs/qwen38-27b-quant-arms.md` and `tests/vllm/models/test_qwen38_27b_gguf_manifest.cpp` both say 48. Introduced by `04ed7b984`. FIXED IN FLOW: all four corrected to 48, and `_stand_in_attn_groups()` now carries the measured 10/4/1 group shape and the real layer names, because a fixture is what the next reader lifts a count from. That alignment closed a real hole rather than a cosmetic one -- turning the per-backend sum into an overwrite leaves the OLD one-group-per-backend fixture GREEN and reds the new one. A FIFTH copy is in the append-only #1658 row and cannot be edited; this row is its correction. Owned by [`dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) `## Owed` O30
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1666
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:613`

### Frozen archive evidence

> | [#1666](https://github.com/mudler/vllm.cpp/issues/1666) | `SPEC-DFLASH2` | **The O26C `attn_groups` census records 30 GDN layers; the run resolved 48, and 30 is the TEST STAND-IN's shape.** Four places -- `.agents/specs/dflash2-spec-decode.md` `## Owed` O30, the `BACKEND_GROUP_PROBES` comment in `tools/bench/dflash2_speed_harness.py`, `resolve_attention_backend_groups` in `tools/bench/dflash2_oracle_capture.py`, and `test_ONE_SCALAR_UNDER_DESCRIBES_this_model_so_the_MAP_is_recorded_too` -- recorded the 2026-08-22 leased walk as `GDNAttentionBackend` over **30** `linear_attn` layers. Re-derived from the run's own `c-probe-result.json`, under `candidate_walks[...worker.model_runner.attn_groups]`, by parsing every `AttentionGroup(backend=..., layer_names=[...])` pair and asserting the match count equals the 15 occurrences of `AttentionGroup(backend=`: **48 GDN layers in 10 groups** (eight of five, two of four), 16 `self_attn.attn` in 4 groups at every 4th index 3-63, and 5 draft layers (`model.layers.64-68`) in 1. The 16 and the 5 were right; only the 30 was wrong, and 30 is `_stand_in_attn_groups()`'s `range(30)` -- a fixture's arbitrary shape generalised into a measurement claim, on the row whose whole history is that failure class. The tree already disagreed with itself: `.agents/specs/qwen38-27b-quant-arms.md` and `tests/vllm/models/test_qwen38_27b_gguf_manifest.cpp` both say 48. Introduced by `04ed7b984`. FIXED IN FLOW: all four corrected to 48, and `_stand_in_attn_groups()` now carries the measured 10/4/1 group shape and the real layer names, because a fixture is what the next reader lifts a count from. That alignment closed a real hole rather than a cosmetic one -- turning the per-backend sum into an overwrite leaves the OLD one-group-per-backend fixture GREEN and reds the new one. A FIFTH copy is in the append-only #1658 row and cannot be edited; this row is its correction. Owned by [`dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) `## Owed` O30 | bug |

## Resolution

-
