ID: ISSUE-GH-863
Title: The strict trailer walk is diff-scoped but ran in `agent-record`, which carries a cancellable group keyed on `github.ref`; on run 31851003245 it was cancelled and `51e0cb5b1` landed failing it with CI silent. A workflow-level cancel takes group-free jobs too, so the real repair is the self-healing base, not the job move (spec [`ci-concurrency.md`](../../specs/ci-concurrency.md))
Row: GATE-CI-CONCURRENCY
State: UNKNOWN
Kind: bug
GitHub: 863
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:219`

### Frozen archive evidence

> | [#863](https://github.com/mudler/vllm.cpp/issues/863) | `GATE-CI-CONCURRENCY` | The strict trailer walk is diff-scoped but ran in `agent-record`, which carries a cancellable group keyed on `github.ref`; on run 31851003245 it was cancelled and `51e0cb5b1` landed failing it with CI silent. A workflow-level cancel takes group-free jobs too, so the real repair is the self-healing base, not the job move (spec [`ci-concurrency.md`](../../specs/ci-concurrency.md)) | bug |

## Resolution

-
