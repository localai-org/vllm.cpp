ID: ISSUE-GH-1838
Title: **The DFlash propose pre-phase ping-pongs the `[T, H*taps]` aux tap through the host (D2H + scalar cast + H2D + D2H) with three hard syncs per step, before the `t_fwd0` timer, so `VT_SPEC_TRACE` cannot even see it.** `exec_state_.spec_aux` is already a device tensor and upstream's `propose` consumes it device-side with no round trip. Fixed in the same W8 flow as #1837 (one reshape of one propose path): `CombineAuxFeaturesDevice` GEMMs the bf16 tap directly, the accepted-prefix gather is a device `IndexSelect` over indices the rejection output already determines, and the context-KV append is fed device-side; the `[spec-phase]` trace gains a `pre=` term so the phase stays attributed. Serves DFlash1 and DFlash2 alike (the pre-phase is the shared seam). Wave spec [dflash2-device-propose.md](../../specs/dflash2-device-propose.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: perf
GitHub: 1838
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:690`

### Frozen archive evidence

> | [#1838](https://github.com/mudler/vllm.cpp/issues/1838) | `SPEC-DFLASH2` | **The DFlash propose pre-phase ping-pongs the `[T, H*taps]` aux tap through the host (D2H + scalar cast + H2D + D2H) with three hard syncs per step, before the `t_fwd0` timer, so `VT_SPEC_TRACE` cannot even see it.** `exec_state_.spec_aux` is already a device tensor and upstream's `propose` consumes it device-side with no round trip. Fixed in the same W8 flow as #1837 (one reshape of one propose path): `CombineAuxFeaturesDevice` GEMMs the bf16 tap directly, the accepted-prefix gather is a device `IndexSelect` over indices the rejection output already determines, and the context-KV append is fed device-side; the `[spec-phase]` trace gains a `pre=` term so the phase stays attributed. Serves DFlash1 and DFlash2 alike (the pre-phase is the shared seam). Wave spec [dflash2-device-propose.md](../../specs/dflash2-device-propose.md) | perf |

## Resolution

-
