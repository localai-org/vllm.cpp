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
| `*.log` | raw leg logs, as run |

## Method

W1 methodology (docs/bench-evidence/flint-cpu-parity-20261029/): per-tensor
rel_rms against the CPU shadow, isolated by staging order.
