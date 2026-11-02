# CLAIM-PASCAL-VOLTA-ENABLE

| Claim | Row IDs | Agent | Worktree / remote dir | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-PASCAL-VOLTA-ENABLE` | `BACKEND-CUDA-SM060` (`ACTIVE`), `BACKEND-CUDA-SM061` (`ACTIVE`), `BACKEND-CUDA-SM070` (`ACTIVE`) | Codex fresh implementer; independent review required | `/tmp/lumen-pascal` | `row/BACKEND-CUDA-SM061` | Portable legacy CUDA implementation, tests, documentation, and these three backend rows. Excludes rebranding, newer-GPU behavior, model downloads, and remote writes. Spec: [Pascal and Volta enablement](../specs/pascal-volta-enable.md). Issue: `ISSUE-GH-146`. | `ACTIVE` | 2026-10-29: production sm_60/61/70 build verified; Pascal runtime regression and public-entry-point gates in progress. Published-checkpoint correctness and performance remain pending. Volta and P100 runtime await hardware. |
