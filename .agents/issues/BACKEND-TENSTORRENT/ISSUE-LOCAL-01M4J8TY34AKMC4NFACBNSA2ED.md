ID: ISSUE-LOCAL-01M4J8TY34AKMC4NFACBNSA2ED
Title: TT kQkvSplit rejects the OPT qkv-split shape and the caller then SIGSEGVs
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-10
Updated: 2026-10-10
Closed: 2026-10-10

## Problem

The P150 model sweep leg 1 (/tmp/p150-leg1.log, OPT-125M bf16, /mnt/models/opt-125m-bf16-st, TT backend, tests/vllm/models/test_opt_paged_engine.cpp:144) fails with 'vt: tenstorrent kQkvSplit: out shapes must be [T, *]' thrown from src/vt/tenstorrent/tenstorrent_ops.cpp:1437 (src/vt/tenstorrent/tenstorrent_ops.cpp:1440 in origin/main); the test harness then SIGSEGVs (exit 139) and no tokens are served. The caller src/vllm/model_executor/models/opt.cpp:132-138 passes rank-3 [T, Hq, Dh] q/k/v outs (mirrors vLLM opt.py qkv.chunk + head reshape). CPU (src/vt/cpu/cpu_ops.cpp:4218), Vulkan (src/vt/vulkan/vulkan_ops.cpp:1635) and Metal QkvSplit all compute the flat widths as Numel/t and are rank-agnostic; only the TT op adds a rank-2 restriction on the outs. The op contract is the cross-backend one, so the TT op must accept rank-2 or rank-3 contiguous outs.

## Resolution

2026-10-10: The caller is right; the TT op's extra rank-2 restriction was the
defect. CPU, Vulkan and Metal QkvSplit all compute the flat widths as Numel/t
and accept rank-2 or rank-3 outs; the TT op now mirrors that contract
(src/vt/tenstorrent/tenstorrent_ops.cpp) and commits its device slices via
CommitDeviceLogical2D (same pattern as tenstorrent_gdn.cpp:72-74). Evidence:
red-first unit test `kTENSTORRENT kQkvSplit accepts rank-3 [T, H, Dh] outs
(OPT shape)` failed on origin/main with the sweep's exact throw, then passed
with the fix; full TT unit suite unchanged vs main (the 2 failures there
reproduce on unpatched main and are unrelated, issues owed separately);
OPT-125M bf16 e2e on device now serves tokens and PASSES the sacred STRICT
gate (6/6 prompts, 96/96 tokens token-exact vs the vLLM 0.25.0 oracle, all 9
OPT ops on device type 6 with 0 declines). Evidence note:
docs/bench-evidence/opt125m-tt-serve-20261010.md. Residue owed: the harness
still SIGSEGVs during teardown after SUCCESS (also visible pre-fix) — a
separate, unattributed defect.
