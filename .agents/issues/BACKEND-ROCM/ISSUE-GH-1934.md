ID: ISSUE-GH-1934
Title: RocmPlatform::needs_weight_staging() is stale-false, so the #1123/#1870 device-fit refusal never runs on ROCm
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: 1934
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-25
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> `RocmPlatform::needs_weight_staging()` ([`rocm.cpp:88`](../../../src/vllm/platforms/rocm.cpp#L88))
> hardcodes `false`, with a comment dated to the W0 skeleton: "in W0 there is one
> registered op, so the only path that can run at all is the host-resident one
> ... a discrete AMD card will eventually answer true ... Revisit at M2."
>
> That comment is stale. Since W0, ROCm has landed grouped/non-grouped keep-quant
> expert GEMM (#523), MoE combine/gate ops (#509), decode-skinny wvSplitK GEMMs
> (#506), self-registered ROCM_ATTN, and hipGraph decode capture (W1). Measured
> on this box (`gfx1200`), `tests/vt/test_backend_cross_device` runs real ROCm
> MoE kernels against the CPU oracle within tolerance today. The platform is not
> "one registered op" anymore.
>
> ## Why this is more than a stale comment
>
> `needs_weight_staging()` gates the ONE production call site of
> `CheckDeviceWeightFit` ([`model_loader.cpp:2273`](../../../src/vllm/entrypoints/model_loader.cpp#L2273)),
> the load-time refusal ENG-EXPERT-STREAM (#1123) added specifically so a
> checkpoint that cannot fit device memory is refused BY NAME before any
> allocation, instead of dying mid-load with a raw `cudaMalloc`/`hipMalloc: out
> of memory`. `CheckDeviceWeightFit` returns immediately when
> `needs_weight_staging()` is false — nothing is even computed.
>
> Measured directly: on this box, loading a checkpoint through `vllm-cli
> --device auto` with `VT_DEVICE_WEIGHT_BUDGET_BYTES=1` (a budget nothing can
> satisfy) produced **no refusal at all** and proceeded to the next load stage.
> Meanwhile the actual device weight allocation this refusal exists to guard —
> `d.b.Alloc(nb)` in `qwen3_5.cpp`'s `ResidentWeight` construction — is **not**
> gated by `needs_weight_staging()`; it allocates unconditionally on whatever
> device the resolved queue targets. So on ROCm today: the crash #1870 reports
> (`vt rocm: hipMalloc: out of memory`) is real and reachable, and the refusal
> meant to replace it with a named message is not, because the guard that would
> run it believes this platform never stages weights.
>
> ## Relationship to #1870
>
> [#1870](https://github.com/mudler/vllm.cpp/issues/1870) reports the crash and
> names the fix as "resolve the expanded residency requirement at load, compare
> it against the device budget, and refuse." That arithmetic fix landed
> separately (`policy_forces_full_expand` in `gguf_device_fit.h`/`.cpp`,
> `.agents/specs/gguf-device-fit-expand-policy.md`) and is necessary — the
> pre-existing `min()`-based bound under-counted a forced-expand load by
> ~4x — but it is **not sufficient**: with `needs_weight_staging()` false, the
> corrected bound is still never evaluated on ROCm. #1870 stays open until this
> row closes the gap.
>
> ## Scope
>
> Not proposed as a design, only to bound it: flipping `needs_weight_staging()`
> (and giving `residency_policy()` a real, probed `device_memory_total_bytes`
> instead of the current default `{}`) is a platform-capability graduation, not
> a one-line flag flip. It is also the ONE input to at least these other
> decisions, all of which currently take the "not staging" branch on ROCm and
> would move:
>
> - `DirectDeviceLoadEligible` ([`qwen3_5_dense_weights.cpp:125`](../../../src/vllm/model_executor/models/qwen3_5_dense_weights.cpp#L125)) — load-path optimization (straight-to-device vs. via host)
> - `IndexedGdnStateIoEnabled` ([`qwen3_5.cpp:3354`](../../../src/vllm/model_executor/models/qwen3_5.cpp#L3354)) — GDN state-cache kernel dispatch; ROCm currently gets the CPU-style row-copy reference path by default *because* this flag is false
> - `MergedGdnBaEnabled` / `MergedGdnQkvzEnabled`'s eligibility checks (`qwen3_5.cpp:3573`, `3881`) — merged-projection GDN kernel eligibility
> - `PackedGdnDecodeEligibility` (`qwen3_5.cpp:3910`) and `PrepareGdnFp8Resident`/`PrepareBf16Resident` (`qwen3_5.cpp:8478`, `8498`) — decode-path prep gating
>
> Each of those needs its own correctness check once the flag moves — several
> change ROCm's *default* kernel dispatch, not just an opt-in path, so a token-exact
> verification against the pinned oracle is needed per affected model, not just a
> build-and-run.
>
> ## History
>
> Found while implementing #1870's arithmetic fix and trying to reproduce the
> crash end-to-end on real `gfx1200` hardware: the corrected bound never fired
> because the guard around it short-circuits first. Filed separately per
> `AGENTS.md`'s owed-row rule — this is a platform-capability change with its own
> review burden, not a slice of the arithmetic fix.
>
> Owned by `BACKEND-ROCM`.
>

## Resolution

-
