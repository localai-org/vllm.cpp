# CLAIM-QWEN38-P150-REBENCH

| Claim | Row IDs | Agent | Worktree / remote dir | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-QWEN38-P150-REBENCH` | `BACKEND-TENSTORRENT-QWEN38-REBENCH` (`ACTIVE`) — spec/issue row `QWEN38-P150-REBENCH` | maki (zai/glm-5-3-flash), fresh implementer | worktree `/tmp/qwen38-rebench`, build `/tmp/build-q38rebench` (CPUs 64-127), bench zone `taskset -c 32-63` | `row/QWEN38-P150-REBENCH` | Owns ONLY the qwen3.8-27B Q4_K_M P150 re-bench at current head (`VT_TT_KEEPQUANT_INT8DOT=1`, governor sweep performance/schedutil, 3 reps each), the comparison vs the 2026-09-14 measurement, evidence under `docs/bench-evidence/qwen38-p150-rebench-20261102/`, and the public docs rows if quotable. No code change. | `IMPLEMENTING` | 2026-11-05 |
