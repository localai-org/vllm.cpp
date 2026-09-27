ID: ISSUE-GH-673
Title: LTX-2.5 prompt-AdaLN: the row's checkpoint-derived evidence is MANUAL and host-local. `LTX2_CHECKPOINT_ROOT` is set by no workflow (`grep -rn CHECKPOINT_ROOT .github/` exits 1 with zero hits, positive control matches in `tests/` and `.agents/`), so CI executes 784 of 9031 assertions — 8.7%, measured 2026-08-15 — of `test_ltx2_video`, at a case count identical in both configurations, and `scripts/measure-ltx2-prompt-adaln.py` is a manual tool no gate invokes. Listed under `## Owed` in [`ltx25-prompt-adaln.md`](../../specs/ltx25-prompt-adaln.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 673
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:230`

### Frozen archive evidence

> | [#673](https://github.com/mudler/vllm.cpp/issues/673) | — | LTX-2.5 prompt-AdaLN: the row's checkpoint-derived evidence is MANUAL and host-local. `LTX2_CHECKPOINT_ROOT` is set by no workflow (`grep -rn CHECKPOINT_ROOT .github/` exits 1 with zero hits, positive control matches in `tests/` and `.agents/`), so CI executes 784 of 9031 assertions — 8.7%, measured 2026-08-15 — of `test_ltx2_video`, at a case count identical in both configurations, and `scripts/measure-ltx2-prompt-adaln.py` is a manual tool no gate invokes. Listed under `## Owed` in [`ltx25-prompt-adaln.md`](../../specs/ltx25-prompt-adaln.md) | bug |

## Resolution

-
