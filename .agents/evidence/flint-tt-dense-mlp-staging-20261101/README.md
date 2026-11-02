# Flint TT dense-MLP staging — per-tensor evidence (2026-11-01)

Row `FLINT-TT-DENSE-MLP-STAGING`. Spec `.agents/specs/flint-tt-dense-mlp-staging.md`.
Worktree `/tmp/flint-p150-attn`, branch `row/FLINT-TT-DENSE-MLP-STAGING`.

## Purpose

Measure the W0 band's allocation-order sensitivity in the dense gate/up/down
staging (`flint_tt.cpp` `MatmulWeight` path) and land the repair that makes
the model-level band order-insensitive.

## Contents

| File | What it is |
|---|---|
| `per-tensor-tables.md` | rel_rms per tensor, L0/L46, orders (a)(b)(c) |
| `attribution.md` | the verdict: staging-order vs upload-path |
| `*.log` | raw leg logs, as run (diagnose legs a–e in
  `docs/bench-evidence/flint-tt-dense-mlp-staging-20261101/`, fix legs f–m
  beside them) |

## Method

W1 methodology (docs/bench-evidence/flint-cpu-parity-20261029/): per-tensor
rel_rms against the CPU shadow, isolated by staging order.

---

# FLINT-TT-DENSE-MLP-STAGING — the slot-serve fix (2026-11-02)

Row `FLINT-TT-DENSE-MLP-STAGING`, fix stage. The diagnose stage's legs a–e
(2026-11-01/02) are unchanged above. This section adds the fix-stage evidence.

## The defect, as reproduced at the diagnose head

`leg-f-repro-at-diagnose-head.log` (env
`VT_FLINT_TT_STAGING_DUMP=1 VT_FLINT_TT_STAGING_ORDER=oproj-scoped`):

- first-token logits rel_rms **4.639e-1** (the healthy legs sit at 2.683e-1);
- L0 `down` rel_rms **1.8059e+0** (sum_tt −5.0376 vs shadow −12.4643),
  reproducing leg-d exactly;
- `down|recomp` (fresh buffers, re-uploaded bytes) and `down|live` both
  reproduce the model's wrong value in F1, and in F2 `down|live` reproduces
  **F1's** wrong value bit-exactly while F2's own down is correct.

## The mechanism (measured, not inferred)

`leg-l-prepost-probes.log` (the diagnose commit's
`VT_FLINT_TT_DOWN_PROBE=1` pre/post-boundary probes) runs THREE consecutive
down matmuls on the SAME input slot in F1:

| matmul | out block | sum |
|---|---|---|
| preprobe (before the boundary) | fresh draw | **−11.865345 (correct)** |
| the model's own down | the recycled `0x…d5900` block | **−5.037615 (wrong)** |
| repprobe (after the model down) | fresh draw | **−11.865345 (correct)** |

Same input bytes, same weight, same queue — only the OUT block differs. The
recycled block `0x…d5900` is a 40960-B pool block serving a 20480-B request:
the #1922 best-fit **borrow**.

The read side is `EnsureHostBytes` (tenstorrent_residency.cpp). It picked the
host write width from the slot's REGISTERED CAPACITY (`s->bytes`, the block's
class — padded by the borrow), not from the shadow's dtype:

```cpp
if (bytes >= n * sizeof(float))  write f32;      // capacity test
else if (bytes >= n * 2)         write bf16;
```

For a bf16 shadow on a borrowed block, `capacity (40960) >= numel*4 (40960)`
is TRUE, so the download wrote **n×4 f32 bytes** into a buffer the caller then
read as **n×2 bf16** — word-halved garbage. An exact-class block (20480 <
40960) takes the bf16 branch and is correct. That is why:

- the poison follows the OUT block's size class, not its tenant history;
- `down|recomp`/`down|live` on fresh exact-class blocks are always correct;
- Synchronize cannot fix it (the write itself is mis-formatted);
- the W0 band moved with ANY functionally neutral DBuf-lifetime edit — every
  such edit shifts WHICH requests borrow.

The capacity heuristic existed to distinguish an f32 shadow (2× the bytes of a
bf16 one) from a padded bf16 slot. The ambiguity is resolved exactly by the
shadow's own dtype, which `EnsureHostBytes` already holds in `dev`.

## The fix

One hunk in `EnsureHostBytes` (`src/vt/tenstorrent/tenstorrent_residency.cpp`,
the non-conv-transposed arm): the host write width is the SHADOW's dtype —
`dev.dtype() == FLOAT32` writes f32, anything else writes bf16 — with the
capacity demoted to a size CHECK (refuse by name when the block cannot hold
the content). No Synchronize, no copy, no serve-path change.

## The permutation matrix (the W0 band under every lifetime order, AFTER)

All four legs, same model artifact, same box, governor `performance` logged in
each log's header, card reset before the first leg of each batch:

| leg | before | after |
|---|---|---|
| landed | 2.683e-1 | **2.683e-1** |
| reversed | 2.683e-1 | **2.683e-1** |
| extended-DBuf | 2.683e-1 | **2.683e-1** |
| oproj-scoped | 7.224e-1 (repro 4.639e-1) | **2.683e-1** |

The spread collapsed to ONE value: the band is allocation-order-INSENSITIVE.

Per-tensor discriminators after the fix (every leg, every dumped layer):

- L0 `down` rel_rms 1.10e-2, `down|recomp` and `down|live` **0.0000e+00**
  (bit-identical to the model's own down) in every leg;
- L46 `sh_silu` rel_rms **1.7655e-2** (was 1.76e+0 — the same borrowed-slot
  poison rode the shared expert's silu too; the fix cured it as well);
- the greedy chain is IDENTICAL across all four legs
  (`[10444 332 269 332 269 269 269 6963]`, 2/8 vs the CPU reference — the
  recorded hard-divergence bias class, unchanged);

## Rejected before the fix

- `leg-g-persistent-retire-attempt.log`: retiring the slot's persistent
  buffer in `MarkScratchAcquired` — did NOT move the band (4.639e-1 unchanged).
  Reverted. Recorded because the claims file's "recycled buffer the program
  still references" hypothesis pointed there; the probe matrix disproved it.

## Standing gates after the fix

- `gate-paged.log` — test_flint_paged **208/208**.
- `gate-gguf.log` — test_flint_gguf 184/184 (the serving gate).
- `gate-forward.log` — test_flint_forward 110/110.
- `gate-attention-probe.log` — 2 cases, 4 assertions, 64/64 per-head slices
  across both window modes (sliding+RoPE, full-without-RoPE).
- The W0 device case: 2/2 cases, 18/18 assertions in every fixed leg.
