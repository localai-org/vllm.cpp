ID: ISSUE-LOCAL-01M5TWFY79FWJN2BDPKYQZ3VXS
Title: Tenstorrent bf16-out matmul staging reads uncorrelated at M=1
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-26
Updated: 2026-10-26
Closed: -

## Problem

The W3 noise map (docs/bench-evidence/tt-matmul-noise-20261025/, test_flint_tt_w3_noise) shows every bf16-OUT arm at M=1 reads uncorrelated output (rel_rms ~1.2 vs host double, all K/N swept) while f32-out at M=1 and bf16-out at M>=5 are clean (rel_rms 3.2e-4 / 1.7e-3 = correct bf16 rounding). The class is in the staging/readback of a [1,N] bf16 result after CommitDevice2D (output side), not the compute config: the W3 arm flip (HiFi4+f32-dest, both arms) does not change it, and the same binary's f32 path through the same CommitDevice2D is bit-clean. Found during W3; the model path uses the f32-out arm, so nothing shipped is reached today. Fix belongs with a staging-layer repair + exactness CHECK at M=1 bf16-out, red-first.

## Resolution

-
