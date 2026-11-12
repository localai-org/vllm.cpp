# Tag identification: the first stock llama.cpp release carrying #30673 and #30865

Read 2026-10-31 through the GitHub API (`api.github.com/repos/ggml-org/llama.cpp`),
cross-checked against a shallow clone of the tag.

## The answer

**Tag `b11765`, object `9df0b73c707900b7277a46b5ae52ff7e7e567d13`.**

`b11765` IS the merge commit of PR #30865 ("model : drop the redundant
precision overrides in the Kolibri-1 router", `merged_at`
2026-10-20T12:58:09Z, `merge_commit_sha` `9df0b73c707900b7277a46b5ae52ff7e7e567d13`
— query `GET /pulls/30865`), so it carries #30865 by construction, and it
carries #30673 because that merge (`62318cd57`, 2026-10-17) is an ancestor:
`GET /compare/62318cd57...b11765` reports `ahead_by 0, behind_by 40` for the
Kolibri-1 merge, i.e. `62318cd57` is contained. No tag between `b11709`
(`62318cd57` itself, the current pin) and `b11765` exists in the tag list, so
`b11765` is the first release tag after the router-precision merge.

Containment evidence, from the compare API (base...head):

| query | status | ahead_by | behind_by |
|---|---|---|---|
| `62318cd57...8642541b` (v0.8.0) | ahead | 251 | 0 |
| `9df0b73c...8642541b` (v0.8.0) | ahead | 195 | 0 |
| `620d2225c...62318cd57` (v0.7.0...kolibri merge) | ahead | 40 | 0 |
| `620d2225c...9df0b73c` (v0.7.0...router fix) | ahead | 96 | 0 |

`behind_by 0` in the first column means the merge commit is an ancestor of the
later object; the v0.7.0 rows show the 2026-10-15 release carries NEITHER merge.

## Why `b11765` and not `v0.8.0`

The repository now carries two tag families: the weekly `b<count>` releases and
a newer semver series (`v0.5.0` 2026-09-23, `v0.6.0` 2026-10-05, `v0.7.0`
2026-10-15, `v0.8.0` 2026-10-28). Both `b11765` and `v0.8.0` carry both merges;
`b11765` is the FIRST release tag that does, which is what the row's spec names
("the first release tag after 2026-10-20"). `v0.8.0` is a valid later
alternative and is recorded here as the fallback if a future advance wants a
wider gap.

## The local mirror

The scoped oracle [`llama-cpp-kolibri1`](../../../.agents/oracles/llama-cpp-kolibri1.md)
already pins `b11765` for the kolibri1-only architecture scope, read through
the API on 22 October 2026; this identification agrees with it independently
(fresh API read + fresh clone) and the advance makes the stock pin and the
scoped pin the same object — see the reconciliation table.

## Clone assertion

```
$ git -C /tmp/lc-v080-src rev-parse HEAD   # after `git fetch --depth 1 origin tag b11765 && git checkout b11765`
9df0b73c707900b7277a46b5ae52ff7e7e567d13
$ git status --porcelain | wc -l
0
```
