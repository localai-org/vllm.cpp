ID: ISSUE-GH-2056
Title: **`check-agent-record.py` accepts TWO claim files owning the same matrix row**, so a claim collision merges clean and silent. Measured on this branch: copying W6a's `CLAIM-MODEL-MM-QWEN4-EXP.md` beside W1's `CLAIM-MODEL-MM-QWEN4-EXP-W1.md` gives `agent record OK`, rc=0, with both files asserting ownership of `MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation` and both marked `ACTIVE`. Git cannot conflict on it because the two sides touch different PATHS. The matrix owner cell holds exactly ONE value, so the record goes silently ambiguous. Resolved here by merge ORDER, which is an operator remembering rather than a gate. NOT fixed in flow: it changes checker semantics and owes its own row, spec and red-before test per AGENTS.md. Listed under `## Owed` in [qwen4-exp-flash-next.md](../../specs/qwen4-exp-flash-next.md).
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: bug
GitHub: 2056
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:787`

### Frozen archive evidence

> | [#2056](https://github.com/mudler/vllm.cpp/issues/2056) | `MODEL-MM-QWEN4-EXP` | **`check-agent-record.py` accepts TWO claim files owning the same matrix row**, so a claim collision merges clean and silent. Measured on this branch: copying W6a's `CLAIM-MODEL-MM-QWEN4-EXP.md` beside W1's `CLAIM-MODEL-MM-QWEN4-EXP-W1.md` gives `agent record OK`, rc=0, with both files asserting ownership of `MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation` and both marked `ACTIVE`. Git cannot conflict on it because the two sides touch different PATHS. The matrix owner cell holds exactly ONE value, so the record goes silently ambiguous. Resolved here by merge ORDER, which is an operator remembering rather than a gate. NOT fixed in flow: it changes checker semantics and owes its own row, spec and red-before test per AGENTS.md. Listed under `## Owed` in [qwen4-exp-flash-next.md](../../specs/qwen4-exp-flash-next.md). | bug |

## Resolution

-
