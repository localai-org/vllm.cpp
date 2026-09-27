ID: ISSUE-GH-1417
Title: **Four record gates return rc 0 on a document carrying literal conflict markers.** Re-derived on a detached scratch worktree at `b537a5344`: five lines spliced into the `docs/STATUS.md` capability table (start marker, a duplicated keyed row, separator, the same row again, end marker) leave `scripts/check-public-doc-tables.py`, `scripts/check-agent-record.py`, `scripts/check-doc-checkpoint.py` and `scripts/check-issue-index-append-only.py` all at rc 0 with their normal OK messages. The two range-scoped checkers were run over a scratch COMMIT, because a working-tree mutation of a commit-reading checker returns 0 without ever reading the mutated bytes. Closed by one tree-scoped checker, `scripts/check-conflict-markers.py`, wired into `scripts/agent-preflight.sh` and the `agent-record` CI job: it refuses a line that starts with seven `<` or seven `>` and a space, and a line of exactly seven `=` only when a start marker opened a hunk above it. The separator stays conditional because a bare row of `=` is a legal setext heading underline, and five shipped files already carry lines of eight or more `=`. No allowlist: the checker builds its patterns from character repetition, so it and its suite carry no marker at column 0. Spec [gate-conflict-markers.md](../../specs/gate-conflict-markers.md)
Row: GATE-CONFLICT-MARKERS
State: UNKNOWN
Kind: bug
GitHub: 1417
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:500`

### Frozen archive evidence

> | [#1417](https://github.com/mudler/vllm.cpp/issues/1417) | `GATE-CONFLICT-MARKERS` | **Four record gates return rc 0 on a document carrying literal conflict markers.** Re-derived on a detached scratch worktree at `b537a5344`: five lines spliced into the `docs/STATUS.md` capability table (start marker, a duplicated keyed row, separator, the same row again, end marker) leave `scripts/check-public-doc-tables.py`, `scripts/check-agent-record.py`, `scripts/check-doc-checkpoint.py` and `scripts/check-issue-index-append-only.py` all at rc 0 with their normal OK messages. The two range-scoped checkers were run over a scratch COMMIT, because a working-tree mutation of a commit-reading checker returns 0 without ever reading the mutated bytes. Closed by one tree-scoped checker, `scripts/check-conflict-markers.py`, wired into `scripts/agent-preflight.sh` and the `agent-record` CI job: it refuses a line that starts with seven `<` or seven `>` and a space, and a line of exactly seven `=` only when a start marker opened a hunk above it. The separator stays conditional because a bare row of `=` is a legal setext heading underline, and five shipped files already carry lines of eight or more `=`. No allowlist: the checker builds its patterns from character repetition, so it and its suite carry no marker at column 0. Spec [gate-conflict-markers.md](../../specs/gate-conflict-markers.md) | bug |

## Resolution

-
