ID: ISSUE-GH-1541
Title: **A REFUSING length guard at the request boundary: the #1365 fix removed the quadratic cost but added no bound.** `SPEC-BPE-QUADRATIC-MERGE` took 64 KB in one pretoken from 23,620.695 ms to 7.797 ms and moved the exponent from ~2 to ~1 (`67823aee2`, [#1539](https://github.com/mudler/vllm.cpp/pull/1539)); it did not add a LIMIT, and a linear cost against a 100 MB body is a smaller problem than a quadratic one rather than the absence of one. The only bound in the stack today is httplib's `CPPHTTPLIB_PAYLOAD_MAX_LENGTH` of 100 MB, and there is still no authentication anywhere in `src/vllm/entrypoints/`. Two binding constraints, both from `.agents/specs/bpe-quadratic-merge.md` `## Defence in depth`: it must REFUSE with an error naming the limit and never truncate, because silently shortening a prompt returns a model output for text the caller did not send; and it belongs at the request boundary rather than in `src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen`, which needs the token count the expensive step produces and so cannot run before it -- placing the guard there reproduces the exact ordering that made the original defect reachable. A byte or character bound is checkable before any tokenization happens, which is the point. NOT a defect #1365 leaves behind and not fixed in that flow: the implementing branch carried no recorded remote-write authority, so its `## Outcome` named the filing as owed AT LANDING and the operator filed it at the merge. Owed under `## Owed` in [bpe-quadratic-merge.md](../../specs/bpe-quadratic-merge.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1541
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:546`

### Frozen archive evidence

> | [#1541](https://github.com/mudler/vllm.cpp/issues/1541) | — | **A REFUSING length guard at the request boundary: the #1365 fix removed the quadratic cost but added no bound.** `SPEC-BPE-QUADRATIC-MERGE` took 64 KB in one pretoken from 23,620.695 ms to 7.797 ms and moved the exponent from ~2 to ~1 (`67823aee2`, [#1539](https://github.com/mudler/vllm.cpp/pull/1539)); it did not add a LIMIT, and a linear cost against a 100 MB body is a smaller problem than a quadratic one rather than the absence of one. The only bound in the stack today is httplib's `CPPHTTPLIB_PAYLOAD_MAX_LENGTH` of 100 MB, and there is still no authentication anywhere in `src/vllm/entrypoints/`. Two binding constraints, both from `.agents/specs/bpe-quadratic-merge.md` `## Defence in depth`: it must REFUSE with an error naming the limit and never truncate, because silently shortening a prompt returns a model output for text the caller did not send; and it belongs at the request boundary rather than in `src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen`, which needs the token count the expensive step produces and so cannot run before it -- placing the guard there reproduces the exact ordering that made the original defect reachable. A byte or character bound is checkable before any tokenization happens, which is the point. NOT a defect #1365 leaves behind and not fixed in that flow: the implementing branch carried no recorded remote-write authority, so its `## Outcome` named the filing as owed AT LANDING and the operator filed it at the merge. Owed under `## Owed` in [bpe-quadratic-merge.md](../../specs/bpe-quadratic-merge.md) | bug |

## Resolution

-
