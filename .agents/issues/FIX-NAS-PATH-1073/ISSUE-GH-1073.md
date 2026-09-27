ID: ISSUE-GH-1073
Title: The NAS moved to `/usr/local/nas_share` and `/mnt/nas_share` is gone, so every tracked default built on `/mnt` broke. `/mnt` is the EPHEMERAL root overlay of the gate box's immutable Kairos OS and does not survive a reboot; `/usr/local` is `COS_PERSISTENT` and does. Observed 2026-08-16 after an 8 h 19 min outage: the mount came back because the `/oem` boot-stage unit worked, `/mnt/nas_share` did not, and the untracked `.env` still declared `CHECKPOINT_ROOT=/mnt/nas_share/checkpoints` — a gate that reads a path `.env` does not declare is not the gate its spec names. `.agents/environment.md` documented NO NAS location at all (measured: the file held no `/mnt` string), so the repair adds the path AND the `COS_PERSISTENT` reason, because a bare path correction invites the next reader to restore the dead location as a symlink. The seven live defaults now derive from `CHECKPOINT_ROOT`, which four sibling scripts already did: `scripts/gen-minimax-music3-manifest.py:17`, `scripts/gen-ltx2-quant-goldens.py:48`, `tools/parity/dump_tokenizer_gpt4o.py:36,39,57`, `tools/gen_pretok_goldens.py:57`, `src/vllm/tokenizer/pretokenizer.cpp:319`, `tests/parity/test_minimax_music3_quant_real.cpp:133,144` and `docs/USAGE.md:3069,3453`. The 41 hits were classified before any edit and the records that cite the old path KEEP it: `.agents/benchmark-record.md`, the LTX-2.5/Nemotron-H specs, `.agents/model-matrix.md`, the captured goldens and the generated `.inc` headers state where a past measurement read its bytes, which is provenance. Spec [`nas-mount-path.md`](../../specs/nas-mount-path.md)
Row: FIX-NAS-PATH-1073
State: UNKNOWN
Kind: bug
GitHub: 1073
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:314`

### Frozen archive evidence

> | [#1073](https://github.com/mudler/vllm.cpp/issues/1073) | `FIX-NAS-PATH-1073` | The NAS moved to `/usr/local/nas_share` and `/mnt/nas_share` is gone, so every tracked default built on `/mnt` broke. `/mnt` is the EPHEMERAL root overlay of the gate box's immutable Kairos OS and does not survive a reboot; `/usr/local` is `COS_PERSISTENT` and does. Observed 2026-08-16 after an 8 h 19 min outage: the mount came back because the `/oem` boot-stage unit worked, `/mnt/nas_share` did not, and the untracked `.env` still declared `CHECKPOINT_ROOT=/mnt/nas_share/checkpoints` — a gate that reads a path `.env` does not declare is not the gate its spec names. `.agents/environment.md` documented NO NAS location at all (measured: the file held no `/mnt` string), so the repair adds the path AND the `COS_PERSISTENT` reason, because a bare path correction invites the next reader to restore the dead location as a symlink. The seven live defaults now derive from `CHECKPOINT_ROOT`, which four sibling scripts already did: `scripts/gen-minimax-music3-manifest.py:17`, `scripts/gen-ltx2-quant-goldens.py:48`, `tools/parity/dump_tokenizer_gpt4o.py:36,39,57`, `tools/gen_pretok_goldens.py:57`, `src/vllm/tokenizer/pretokenizer.cpp:319`, `tests/parity/test_minimax_music3_quant_real.cpp:133,144` and `docs/USAGE.md:3069,3453`. The 41 hits were classified before any edit and the records that cite the old path KEEP it: `.agents/benchmark-record.md`, the LTX-2.5/Nemotron-H specs, `.agents/model-matrix.md`, the captured goldens and the generated `.inc` headers state where a past measurement read its bytes, which is provenance. Spec [`nas-mount-path.md`](../../specs/nas-mount-path.md) | bug |

## Resolution

-
