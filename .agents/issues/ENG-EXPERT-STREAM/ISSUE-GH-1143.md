ID: ISSUE-GH-1143
Title: `src/vllm/entrypoints/model_loader.cpp` is cited by ABSOLUTE LINE NUMBER from **109 distinct sites across 45 files** — specs, matrices, docs and comments in other translation units — and the file is ~1640 lines that almost every engine and model row edits. Any edit near its top invalidates every citation below it, in files the editing change never opens. Measured on [#1136](https://github.com/mudler/vllm.cpp/issues/1136)'s repair, which inserts ~45 lines near line 100 (`AutoDeviceResolution` / `ResolveAutoDevice`, so the device resolver and the queue selector share one description): comparing the TEXT at every cited line between `e7d0a1f7c` and the repaired head gives **203 moved line references over 109 citing sites in 45 files, 10 unmoved**. What that number is NOT: a claim that 109 correct citations broke. Several were already stale — `.agents/model-matrix.md:197` cites `model_loader.cpp:184-223` as the "live loader" while line 184 at `e7d0a1f7c` is `static const bool once = [] {` inside the `VT_LOAD_STATS` helper. The finding is that the surface cannot survive an ordinary edit and that nobody can currently tell the two cases apart. NOT swept there, for two reasons: it is 109 sites in specs owned by other rows, and rewriting them all from the current tree would launder pre-existing debt into a clean-looking record. What #1136 DID fix is the two anchors it authored itself, plus adding them to its own anchor verifier so they could not go stale inside their own pull request — the defect the round before hit with `platforms/cuda.cpp:67`. Fix candidates, none chosen: cite SYMBOLS not lines across file boundaries (the only one that removes the class); an anchor gate holding each `file:line` against an expected substring (a checker change, and its expectation table is itself a shared-file lock); or split the 1640-line file, whose size is what makes the blast radius large. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md) until a row claims it
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1143
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:348`

### Frozen archive evidence

> | [#1143](https://github.com/mudler/vllm.cpp/issues/1143) | `ENG-EXPERT-STREAM` | `src/vllm/entrypoints/model_loader.cpp` is cited by ABSOLUTE LINE NUMBER from **109 distinct sites across 45 files** — specs, matrices, docs and comments in other translation units — and the file is ~1640 lines that almost every engine and model row edits. Any edit near its top invalidates every citation below it, in files the editing change never opens. Measured on [#1136](https://github.com/mudler/vllm.cpp/issues/1136)'s repair, which inserts ~45 lines near line 100 (`AutoDeviceResolution` / `ResolveAutoDevice`, so the device resolver and the queue selector share one description): comparing the TEXT at every cited line between `e7d0a1f7c` and the repaired head gives **203 moved line references over 109 citing sites in 45 files, 10 unmoved**. What that number is NOT: a claim that 109 correct citations broke. Several were already stale — `.agents/model-matrix.md:197` cites `model_loader.cpp:184-223` as the "live loader" while line 184 at `e7d0a1f7c` is `static const bool once = [] {` inside the `VT_LOAD_STATS` helper. The finding is that the surface cannot survive an ordinary edit and that nobody can currently tell the two cases apart. NOT swept there, for two reasons: it is 109 sites in specs owned by other rows, and rewriting them all from the current tree would launder pre-existing debt into a clean-looking record. What #1136 DID fix is the two anchors it authored itself, plus adding them to its own anchor verifier so they could not go stale inside their own pull request — the defect the round before hit with `platforms/cuda.cpp:67`. Fix candidates, none chosen: cite SYMBOLS not lines across file boundaries (the only one that removes the class); an anchor gate holding each `file:line` against an expected substring (a checker change, and its expectation table is itself a shared-file lock); or split the 1640-line file, whose size is what makes the blast radius large. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md) until a row claims it | bug |

## Resolution

-
