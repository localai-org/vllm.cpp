ID: ISSUE-GH-595
Title: `check-doc-checkpoint` keys `feature_surface` off the PATH `src/vllm/model_executor/models/`, so every edit to any model TU owes `docs/FEATURES.md` — the same classify-by-directory defect the file's own header says its rewrite removed for `src/`, `include/` and `tests/`. Measured cost: `e34d71379` (#1054) is a one-line lambda-capture change that alters no capability; the gate demanded the surface, the commit answered with prose, the prose crossed the `check-public-doc-tables` paragraph budgets, and because that checker also runs in the pre-push hook it blocked EVERY branch in the repository from pushing ([#1055](https://github.com/mudler/vllm.cpp/issues/1055), re-filed as [#1062](https://github.com/mudler/vllm.cpp/issues/1062) with a duplicate fix PR, plus [#1058](https://github.com/mudler/vllm.cpp/issues/1058) still open). The repair for the MSVC break the same commit caused ([#1068](https://github.com/mudler/vllm.cpp/issues/1068)) hit the identical demand. Narrowed here to a change in the set of `REGISTER_VLLM_MODEL(...)` registrations, which is what `check-supported-models.py` already gates the table against; adding, removing or renaming an architecture still owes the surface. The LOCK this issue names is NOT closed by that — a genuine new architecture still writes the shared table — so #595 stays open, listed under `## Owed` in [`doc-checkpoint-feature-trigger.md`](../../specs/doc-checkpoint-feature-trigger.md). Sibling shape for `CMakeLists.txt` -> `docs/USAGE.md` is [#515](https://github.com/mudler/vllm.cpp/issues/515)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 595
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:327`

### Frozen archive evidence

> | [#595](https://github.com/mudler/vllm.cpp/issues/595) | — | `check-doc-checkpoint` keys `feature_surface` off the PATH `src/vllm/model_executor/models/`, so every edit to any model TU owes `docs/FEATURES.md` — the same classify-by-directory defect the file's own header says its rewrite removed for `src/`, `include/` and `tests/`. Measured cost: `e34d71379` (#1054) is a one-line lambda-capture change that alters no capability; the gate demanded the surface, the commit answered with prose, the prose crossed the `check-public-doc-tables` paragraph budgets, and because that checker also runs in the pre-push hook it blocked EVERY branch in the repository from pushing ([#1055](https://github.com/mudler/vllm.cpp/issues/1055), re-filed as [#1062](https://github.com/mudler/vllm.cpp/issues/1062) with a duplicate fix PR, plus [#1058](https://github.com/mudler/vllm.cpp/issues/1058) still open). The repair for the MSVC break the same commit caused ([#1068](https://github.com/mudler/vllm.cpp/issues/1068)) hit the identical demand. Narrowed here to a change in the set of `REGISTER_VLLM_MODEL(...)` registrations, which is what `check-supported-models.py` already gates the table against; adding, removing or renaming an architecture still owes the surface. The LOCK this issue names is NOT closed by that — a genuine new architecture still writes the shared table — so #595 stays open, listed under `## Owed` in [`doc-checkpoint-feature-trigger.md`](../../specs/doc-checkpoint-feature-trigger.md). Sibling shape for `CMakeLists.txt` -> `docs/USAGE.md` is [#515](https://github.com/mudler/vllm.cpp/issues/515) | bug |

## Resolution

-
