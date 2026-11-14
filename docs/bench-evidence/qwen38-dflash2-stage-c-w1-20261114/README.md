# W1 evidence — QWEN38-DFLASH2-STAGE-C (the drafter weights), 2026-11-14

Wave W1 of row `QWEN38-DFLASH2-STAGE-C`: enumerate, download, validate, pin
the DFlash2 drafter weights.

- Files: FLAT. `enumeration.json` (all 81 tensors, names/shapes/dtypes/bytes +
  source pin), `validator.log` (the contract run that wrote
  `/mnt/models/qwen38-dflash2-drafter/meta.json`).
- Where the tensors live: `incoai/Qwen3.8-27B-DFlash2` @
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, one flat
  `model.safetensors` (3,848,817,896 B, sha256
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`).
  Provenance: the dflash2-vision bundle `PROVENANCE.json` `weights.drafter`
  (`mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv` @ `28add9a1…`); the
  bundle's drafter code (`extra_code/qwen38/dflash2.py:39`) names the same
  repo as `DEFAULT_WEIGHTS`. Download authorized by the developer
  2026-11-13/14.
- Format decision (recorded): NO re-encode. The loader's existing draft path
  (`LoadDflashDraft`, `src/vllm/entrypoints/model_loader.cpp:512+`;
  contract documented at `include/vllm/model_executor/models/qwen3_dflash.h:14-111`)
  consumes this exact HF layout, and its geometry matches this arm (hidden
  5120, 5 layers, fc [5120, 25600], conv kernel_projection (1280, 5120),
  selector hidden_projection (256, 5120), per-head qk-norm [128]).
  `scripts/convert-qwen38-dflash2-drafter.py` validates every tensor against
  that contract and refuses BY NAME; the conversion is the identity, and the
  script writes `meta.json` (enumeration + pin) beside the checkpoint.
- Refuse-by-name (W1 deliverable): already loader-side —
  `src/vllm/model_executor/models/qwen3_dflash_weights.cpp` refuses
  "tensor not found: <name>" and `refuse_unowned` names the consumer. No W1
  source change to the loader was needed; W2 wires the forward through the
  existing entry point and inherits the refusal.
- Enumerated tensor groups (all BF16, 81 tensors, 3.58 GiB):
  fc.weight [5120,25600] x1; hidden_norm/norm [5120] x2; per layer (x5):
  in/post LN [5120] x2, q [4096,5120], k/v [1024,5120], o [5120,4096],
  q_norm/k_norm [128] x2, conv base_kernel (2,2,5120) +
  kernel_projection (1280,5120) x2 (attention_conv, mlp_conv), mlp
  gate/up [17408,5120], down [5120,17408]; selector codebooks (248320,256)
  x2 + hidden_projection (256,5120).
- One validator-contract correction during bring-up: the script initially
  expected q_norm/k_norm [4096]/[1024] and REFUSED the checkpoint by name;
  the loader contract is per-head [head_dim] = [128]
  (`qwen3_dflash.h:158-159`), so the checkpoint was right and the script's
  table was fixed.
