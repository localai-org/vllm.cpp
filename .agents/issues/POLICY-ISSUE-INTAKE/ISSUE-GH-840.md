ID: ISSUE-GH-840
Title: The issue intake table sits inside `roadmap_v1.md`, which 51 of the last 60 commits touch, and two branches appending a row conflict under the default merge and merge cleanly under `merge=union`; a `.gitattributes` entry binds a path and never a section, so the table moves to `.agents/issue-index.md` and becomes append-only, and ownership becomes a network-free gate because 33 of the 185 rows name no owning row (spec [`issue-intake.md`](../../specs/issue-intake.md))
Row: POLICY-ISSUE-INTAKE
State: UNKNOWN
Kind: bug
GitHub: 840
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:204`

### Frozen archive evidence

> | [#840](https://github.com/mudler/vllm.cpp/issues/840) | `POLICY-ISSUE-INTAKE` | The issue intake table sits inside `roadmap_v1.md`, which 51 of the last 60 commits touch, and two branches appending a row conflict under the default merge and merge cleanly under `merge=union`; a `.gitattributes` entry binds a path and never a section, so the table moves to `.agents/issue-index.md` and becomes append-only, and ownership becomes a network-free gate because 33 of the 185 rows name no owning row (spec [`issue-intake.md`](../../specs/issue-intake.md)) | bug |

## Resolution

-
