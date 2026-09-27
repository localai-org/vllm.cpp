ID: ISSUE-GH-1824
Title: **A configured speculator forces synchronous scheduling; upstream keeps async ON for the Eagle-type family (dflash included).** `model_loader.cpp` resolves `async_scheduling_enabled_` to false whenever `resolved_spec_config_` is present, a deferral recorded at SPEC-MTP I5d ([mtp-spec-decode.md](../../specs/mtp-spec-decode.md) §2.8's not-ported list). Upstream's polarity at the pin (`vllm/config/vllm.py:1064-1112`) disables async only for a method OUTSIDE `EagleModelTypes ∪ NgramGPUTypes ∪ {"dspark"}` — and `"dflash"` (which DFlash2 rides) and every MTP type are Eagle-type. At c1 spec decode (~360 steps / 2048 tokens) every host-side scheduling cost is serialized into each step, the largest named host-side divergence in the [#1574](https://github.com/mudler/vllm.cpp/issues/1574) gap. W7 under `SPEC-DFLASH2` ports the draft-in-output flow (AsyncScheduler `-1` placeholders, worker-side fill, `update_draft_token_ids_in_output`, the `async_tokens_to_discard` rollback guard) and flips the enable to upstream's method predicate; the GPU TPOT A/B stays owed to the operator. Spec [`spec-decode-async-scheduling.md`](../../specs/spec-decode-async-scheduling.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: feature
GitHub: 1824
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:685`

### Frozen archive evidence

> | [#1824](https://github.com/mudler/vllm.cpp/issues/1824) | `SPEC-DFLASH2` | **A configured speculator forces synchronous scheduling; upstream keeps async ON for the Eagle-type family (dflash included).** `model_loader.cpp` resolves `async_scheduling_enabled_` to false whenever `resolved_spec_config_` is present, a deferral recorded at SPEC-MTP I5d ([mtp-spec-decode.md](../../specs/mtp-spec-decode.md) §2.8's not-ported list). Upstream's polarity at the pin (`vllm/config/vllm.py:1064-1112`) disables async only for a method OUTSIDE `EagleModelTypes ∪ NgramGPUTypes ∪ {"dspark"}` — and `"dflash"` (which DFlash2 rides) and every MTP type are Eagle-type. At c1 spec decode (~360 steps / 2048 tokens) every host-side scheduling cost is serialized into each step, the largest named host-side divergence in the [#1574](https://github.com/mudler/vllm.cpp/issues/1574) gap. W7 under `SPEC-DFLASH2` ports the draft-in-output flow (AsyncScheduler `-1` placeholders, worker-side fill, `update_draft_token_ids_in_output`, the `async_tokens_to_discard` rollback guard) and flips the enable to upstream's method predicate; the GPU TPOT A/B stays owed to the operator. Spec [`spec-decode-async-scheduling.md`](../../specs/spec-decode-async-scheduling.md) | feature |

## Resolution

-
