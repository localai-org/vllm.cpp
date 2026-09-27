ID: ISSUE-GH-822
Title: Superseded runs accumulate: 32 obsolete queued runs cancelled by hand on 2026-08-14, and `containers.yml` had no concurrency policy at all. The push lane becomes latest-only, a closed pull request supersedes its own run without gating, and the diff-scoped gates base on the last successfully gated commit so cancellation is lossless (spec [`ci-concurrency.md`](../../specs/ci-concurrency.md))
Row: GATE-CI-CONCURRENCY
State: UNKNOWN
Kind: bug
GitHub: 822
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:218`

### Frozen archive evidence

> | [#822](https://github.com/mudler/vllm.cpp/issues/822) | `GATE-CI-CONCURRENCY` | Superseded runs accumulate: 32 obsolete queued runs cancelled by hand on 2026-08-14, and `containers.yml` had no concurrency policy at all. The push lane becomes latest-only, a closed pull request supersedes its own run without gating, and the diff-scoped gates base on the last successfully gated commit so cancellation is lossless (spec [`ci-concurrency.md`](../../specs/ci-concurrency.md)) | bug |

## Resolution

-
