# Cached RoPE matrix review repair

Base: `6c785c4a8`. Spec-first commit: `035fc6692`.
Issue: ISSUE-LOCAL-01M645FSTTEMK6E7QHQ183ZD3F.

The cached RoPE test now preserves the pinned vLLM F32 and BF16 dtype matrix,
with the declared F16 extension. The 288 dtype/shape combinations retain both
conventions, optional keys, head strides, positions, token counts, and tolerances.
Pinned source: vLLM `a7c23ac96d7806e7c7e7d862eadbce5a33529b94`,
`tests/kernels/core/test_rotary_embedding.py:29-93`.

The upstream alternate flattened layout is materialized and reshaped with a
rank-3 tensor view. The registered local rank contract remains unchanged.
The additional 144 flattened/view calls check logical values after materialization.
The existing strided cases separately check untouched padding and head strides.

Only the stale gamma-only comment changes in `cuda_ops.cu`. Removing comments
makes the old and new source byte-identical. Both versions contain 4253 lines.
No numerical implementation changes belong to this repair.

## Verification

- Full scalar suite: 14 cases, 3428 assertions, exit 0.
- Existing CUDA operation suite: 14 cases, 451 assertions, exit 0.
- Initial restored dtype matrix: 1152 assertions, exit 0.
- Final cached matrix including flattened views: 1728 assertions, exit 0.
- Remove only BF16 cached production dispatch: old committed test exits 0;
  the new matrix exits 1. This proves the reviewed coverage gap.
- Independently remove F32 and F16 cached production dispatch: each exits 1.
- Restore unchanged production objects: final cached matrix exits 0.
- Actual changed CUDA translation unit compiles for sm60, sm61, and sm70.
- Canonical record validation and whitespace checks pass.

All GPU runs hold `/home/user/gpu.lock`. Each mutation uses a separate external
source and executable. Tracked production files are never mutated by these runs.
The supplied recipe reuses the original independent-review thin archive and
immutable objects; binary hashes identify the exact executed artifacts.

The inherited startup preflight was invoked before edits and remains incomplete.
The original implementation's full preflight is reported separately. This scoped
repair does not rerun that long suite or claim a green full preflight.
Fresh scoped review and the operator rerun remain pending. Actual model dtype
wiring, primary model execution, and final model acceptance remain owned by Wave4.
