# Restore the cached RoPE upstream dtype matrix

## Now

IMPLEMENTED. Repair of independent review P2 and P3 at immutable6c785c4a8.
Fresh scoped review and operator acceptance remain pending.
Issue: ISSUE-LOCAL-01M645FSTTEMK6E7QHQ183ZD3F.
Parent design: legacy-cuda-fp16-dense.md. Row: BACKEND-CUDA-SM061.

## Scope and source

Extend only tests/vt/test_cuda_fp16_scalar.cpp cached RoPE matrix and its evidence.
Correct only the stale gamma-only comment in src/vt/cuda/cuda_ops.cu.
Pinned vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94,
tests/kernels/core/test_rotary_embedding.py:29-78 enumerates F32/BF16,
max positions11/4096/32768,both conventions,rotary32,head32/108,
sequence11/1024,key on/off,head stride variants,num heads7,base10000.
Keep all192 original combinations and96 additional F16 combinations.
The upstream flattened alternate shape is adapted by a rank3 reshape/view;
the registered operation's rank ABI does not change. Document this adaptation.
Preserve numerical implementation, existing tolerances and all other files.

## Design and risks

Parameterize input/cache/output packing and expected rounding by dtype.
F32 and BF16 preserve their executing scalar contract. The rank3 view must
preserve the underlying head stride and untouched tail values. A passing F16
case alone cannot detect deleted original-mode dispatch.
Preserve source line count in the comment correction to retain existing anchors.
Do not broaden product rank handling or repair unrelated findings.

## Gates and evidence

Capture old committed cached test PASS with only BF16 production branch removed.
Require the extended test to FAIL under the same mutation. Delete F32 and F16
branches independently and require FAIL. Mutations use external sources; restore
tracked bytes unchanged. Run all288 cached combinations, full scalar suite,
existing operation controls, record/style/whitespace checks. Compile the unchanged
numerical CUDA translation unit for sm60/sm61/sm70 if required by the gate.
No duplicate27-minute preflight is required for this test/comment repair;
record the original full-gate result separately when available. Independent
fresh scoped review and operator rerun remain required.

## Owed

Fresh review and operator acceptance remain required. Parent Wave4 owns staged
automatic model wiring; this test repair supplies no model/runtime acceptance.

## Implementation evidence

See [scoped repair evidence](../evidence/legacy-cuda-fp16-wave1-rope-matrix/README.md).
The full scalar suite passes 14 cases and 3428 assertions. Each dtype call-site
mutation fails the new matrix; the old BF16 omission remains demonstrated.
No product numerical source changes or model acceptance are claimed.
