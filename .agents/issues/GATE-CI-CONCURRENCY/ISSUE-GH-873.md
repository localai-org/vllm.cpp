ID: ISSUE-GH-873
Title: `main` went RED on six release/registration gates after the #865 `ci.yml` rewrite: `check-release-binary-contract.py` and `check-test-registration.py` credit a checker to CI only through an UNCONDITIONAL job, and #865 gave `agent-record` an `if:`; the closed-PR skip is re-expressed through `needs:` and the byte-exact Windows PR proof schema restored (spec [`ci-concurrency.md`](../../specs/ci-concurrency.md))
Row: GATE-CI-CONCURRENCY
State: UNKNOWN
Kind: bug
GitHub: 873
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:257`

### Frozen archive evidence

> | [#873](https://github.com/mudler/vllm.cpp/issues/873) | `GATE-CI-CONCURRENCY` | `main` went RED on six release/registration gates after the #865 `ci.yml` rewrite: `check-release-binary-contract.py` and `check-test-registration.py` credit a checker to CI only through an UNCONDITIONAL job, and #865 gave `agent-record` an `if:`; the closed-PR skip is re-expressed through `needs:` and the byte-exact Windows PR proof schema restored (spec [`ci-concurrency.md`](../../specs/ci-concurrency.md)) | bug |

## Resolution

-
