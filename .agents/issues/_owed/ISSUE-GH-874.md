ID: ISSUE-GH-874
Title: `windows-msvc-cpu`/`windows-msvc-vulkan` still start on a CLOSED pull request: `check-release-workflow.py::validate_pr_ci` compares their whole job mapping for equality, so neither an `if:` clause nor a `needs:` guard can be added. Listed under `## Owed` in [`ci-concurrency.md`](../../specs/ci-concurrency.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 874
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:258`

### Frozen archive evidence

> | [#874](https://github.com/mudler/vllm.cpp/issues/874) | — | `windows-msvc-cpu`/`windows-msvc-vulkan` still start on a CLOSED pull request: `check-release-workflow.py::validate_pr_ci` compares their whole job mapping for equality, so neither an `if:` clause nor a `needs:` guard can be added. Listed under `## Owed` in [`ci-concurrency.md`](../../specs/ci-concurrency.md) | bug |

## Resolution

-
