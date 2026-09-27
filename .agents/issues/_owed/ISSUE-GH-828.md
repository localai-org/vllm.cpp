ID: ISSUE-GH-828
Title: `check-device-leakage.py`'s `dev_cast` bucket enforces a set of SPELLINGS, not the property "an integer becomes a `vt::DeviceType`": four review rounds each found a spelling the previous round's message already claimed, every one closing at ZERO hits, so widening is not the answer. The structural fix is an AST-level check (clang tooling), where the destination type canonicalises and the source type is known. Listed under `## Owed` in [`ltx25-device-seam-sibling.md`](../../specs/ltx25-device-seam-sibling.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 828
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:229`

### Frozen archive evidence

> | [#828](https://github.com/mudler/vllm.cpp/issues/828) | — | `check-device-leakage.py`'s `dev_cast` bucket enforces a set of SPELLINGS, not the property "an integer becomes a `vt::DeviceType`": four review rounds each found a spelling the previous round's message already claimed, every one closing at ZERO hits, so widening is not the answer. The structural fix is an AST-level check (clang tooling), where the destination type canonicalises and the source type is known. Listed under `## Owed` in [`ltx25-device-seam-sibling.md`](../../specs/ltx25-device-seam-sibling.md) | feature |

## Resolution

-
