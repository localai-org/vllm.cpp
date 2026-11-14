ID: ISSUE-LOCAL-01M73J3QJ5CEBP2YS3ESK2ZMSE
Title: Stage C: port the DFlash2 block-diffusion drafter and wire the speculative loop through the captured verify arm
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-10
Updated: 2026-11-14
Closed: -

## Problem

Stage B's captured verify arm is speculative-ready but nothing drafts. The bundle's DFlash2 drafter (extra_code/qwen38/dflash2.py, DFlash2Draft/DFlash2Drafter: one 5-layer forward over the target's residual taps producing 8 fixed-shape rows [anchor, MASK x7]) is the missing producer; the modeled win at the bundle's acceptance is ~8.5x on the 6606.84 ms stage-B arm. Owed: W1 obtain/convert/pin the drafter weights (download needs developer go; converter + docs/USAGE.md pin + refuse-by-name), W2 component-validated drafter forward on TT (PCC band vs the bundle reference), W3 the loop through Qwen3_5DenseDecodeGraph (draft -> captured verify replay -> host greedy accept by token-id compare; reject rolls back via the stage-B GDN commit-or-restore; dense driver only), W4 the gate (greedy token identity vs plain decode, STOP on divergence; acceptance/TPOT table vs the stage-B baseline; the host-staging cost the modeled win assumed away must be attributed honestly). Spec: .agents/specs/qwen38-dflash2-stage-c.md

## Resolution

- 2026-11-14 (W1, weights): the drafter tensors ARE buyable — the bundle's
  `PROVENANCE.json` `weights.drafter` names `incoai/Qwen3.8-27B-DFlash2` @
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, the same repo the bundle's
  drafter code resolves (`extra_code/qwen38/dflash2.py:39`). Downloaded
  (developer go given 2026-11-13/14) to
  `/mnt/models/qwen38-dflash2-drafter/`: `model.safetensors`
  3,848,817,896 B, sha256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`;
  `config.json` 1,239 B sha256 `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`.
  Enumeration: 81 tensors, all BF16 (fc [5120,25600]; hidden_norm/norm [5120];
  5 layers of q/k/v/o + per-head qk-norm [128] + conv base_kernel (2,2,5120)
  + kernel_projection (1280,5120) x2 + MLP 17408; selector codebooks
  (248320,256) + hidden_projection (256,5120)). Format decision recorded: NO
  re-encode — the existing `LoadDflashDraft` draft path consumes this HF
  layout directly (contract `include/vllm/model_executor/models/qwen3_dflash.h:14-111`);
  `scripts/convert-qwen38-dflash2-drafter.py` validates every tensor against
  it, refuses by name, writes `meta.json`. Refuse-by-name at load time already
  exists loader-side (`qwen3_dflash_weights.cpp` "tensor not found" /
  `refuse_unowned`); no W1 loader change owed. Pins: `docs/USAGE.md`
  ("The Qwen3.8 DFlash2 drafter: the exact weights"); evidence
  `docs/bench-evidence/qwen38-dflash2-stage-c-w1-20261114/` (FLAT:
  enumeration.json, validator.log, README.md). W2 (the forward,
  component-validated) is next.

