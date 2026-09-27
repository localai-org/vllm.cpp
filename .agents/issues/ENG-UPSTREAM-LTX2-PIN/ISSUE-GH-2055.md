ID: ISSUE-GH-2055
Title: `tools/oracle/ltx2_oracle.py` asserts the LTX-2 oracle's revision and its resolved `ltx_core` / `ltx_pipelines` origins in the PARENT, then renders in a child started with `python -m`, which puts the CURRENT WORKING DIRECTORY on that child's `sys.path[0]`. The parent's `importlib.util.find_spec` never consults it, so a directory holding a decoy `ltx_pipelines`, made the CWD, is imported by the process that loads the weights while the process that checked identity sees nothing. MEASURED by a fresh reviewer of [#2053](https://github.com/mudler/vllm.cpp/pull/2053) on a scratch copy: the parent printed `IDENTITY_OK` naming the pinned tree and the child then printed `DECOY ti2vid_one_stage RUNNING -- the parent's assert never saw me`. This is the `.agents/specs/ltx-2-5.md` §7.0(b) decoy failure surviving in the one process that touches weights. The 2026-08-27 reference render is NOT affected and its `gateable = yes` stands: `render.sh` issues no `cd`, `/workspace/ltx2-oracle/` holds no `ltx_*` package, and the committed manifest records module origins inside the pinned clone — so the defect is an OVER-CLAIMED guarantee, not a wrong measurement. Fix: `-P` (or `PYTHONSAFEPATH=1`) on the child plus an explicit `cwd=`, with the reviewer's decoy as the red-first test. NOT fixed in flow and deliberately: the script's sha256 equals the one the worker executed and printed, and that equality is the provenance chain `.agents/oracles/ltx-2.md`'s `gateable = yes` rests on, so editing the file for a hardening that changed no result would trade a verifiable fact for a better comment. Owned by `ENG-UPSTREAM-LTX2-PIN` and listed under `## Owed` in [oracle-ltx-2-pin.md](../../specs/oracle-ltx-2-pin.md)
Row: ENG-UPSTREAM-LTX2-PIN
State: UNKNOWN
Kind: bug
GitHub: 2055
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:786`

### Frozen archive evidence

> | [#2055](https://github.com/mudler/vllm.cpp/issues/2055) | `ENG-UPSTREAM-LTX2-PIN` | `tools/oracle/ltx2_oracle.py` asserts the LTX-2 oracle's revision and its resolved `ltx_core` / `ltx_pipelines` origins in the PARENT, then renders in a child started with `python -m`, which puts the CURRENT WORKING DIRECTORY on that child's `sys.path[0]`. The parent's `importlib.util.find_spec` never consults it, so a directory holding a decoy `ltx_pipelines`, made the CWD, is imported by the process that loads the weights while the process that checked identity sees nothing. MEASURED by a fresh reviewer of [#2053](https://github.com/mudler/vllm.cpp/pull/2053) on a scratch copy: the parent printed `IDENTITY_OK` naming the pinned tree and the child then printed `DECOY ti2vid_one_stage RUNNING -- the parent's assert never saw me`. This is the `.agents/specs/ltx-2-5.md` §7.0(b) decoy failure surviving in the one process that touches weights. The 2026-08-27 reference render is NOT affected and its `gateable = yes` stands: `render.sh` issues no `cd`, `/workspace/ltx2-oracle/` holds no `ltx_*` package, and the committed manifest records module origins inside the pinned clone — so the defect is an OVER-CLAIMED guarantee, not a wrong measurement. Fix: `-P` (or `PYTHONSAFEPATH=1`) on the child plus an explicit `cwd=`, with the reviewer's decoy as the red-first test. NOT fixed in flow and deliberately: the script's sha256 equals the one the worker executed and printed, and that equality is the provenance chain `.agents/oracles/ltx-2.md`'s `gateable = yes` rests on, so editing the file for a hardening that changed no result would trade a verifiable fact for a better comment. Owned by `ENG-UPSTREAM-LTX2-PIN` and listed under `## Owed` in [oracle-ltx-2-pin.md](../../specs/oracle-ltx-2-pin.md) | bug |

## Resolution

-
