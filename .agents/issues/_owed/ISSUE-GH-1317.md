ID: ISSUE-GH-1317
Title: The full-model LTX-2.5 render harness compares **absolute** peaks of a **system-wide** column across runs that started 26.812 GiB apart, and [#1286](https://github.com/mudler/vllm.cpp/issues/1286)'s entire reported regression is that offset. `runguard.py:236-237,260` writes `used_gib = MemTotal - MemAvailable` and `anon_gib = AnonPages`, both system-wide; only `rss_gib` belongs to the child. The pre-#1252 run began at `used = 4.741 GiB` on a box with 114.890 GiB available; the #1252 run began at `used = 31.553 GiB` with 88.078 GiB available. Peak minus each run's OWN `t=0` is **74.465 GiB before and 74.300 GiB after**, with system `AnonPages` deltas of 38.012 vs 38.217 GiB and child `VmRSS` at each peak sample of 41.952 vs 42.090 GiB — three instruments inside ±0.25 GiB, both peaks sampled inside a ~1900% CPU stretch rather than in different phases. On a box as clean as the first run's the second binary's own demand would have left **40.59 GiB** available, above both the 12 GiB hard floor and the 8 GiB projection floor. **Two fixes, neither in this repository:** record `used_gib` at `t=0` in `PROVENANCE` and make every cross-run claim on the delta, on `rss_gib`, or refuse it; and find what held 26.8 GiB on `dgx:gpu0` before the job started — a leaked container or orphaned process from a previous job is the candidate, and it is what actually aborted the render. NOT fixed in flow: `runguard.py` and fleet job hygiene are outside the tree, and the second half needs a look at the box rather than a diff. Owed under `## Owed` of [`ltx25-text-linear-mem.md`](../../specs/ltx25-text-linear-mem.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1317
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:429`

### Frozen archive evidence

> | [#1317](https://github.com/mudler/vllm.cpp/issues/1317) | — | The full-model LTX-2.5 render harness compares **absolute** peaks of a **system-wide** column across runs that started 26.812 GiB apart, and [#1286](https://github.com/mudler/vllm.cpp/issues/1286)'s entire reported regression is that offset. `runguard.py:236-237,260` writes `used_gib = MemTotal - MemAvailable` and `anon_gib = AnonPages`, both system-wide; only `rss_gib` belongs to the child. The pre-#1252 run began at `used = 4.741 GiB` on a box with 114.890 GiB available; the #1252 run began at `used = 31.553 GiB` with 88.078 GiB available. Peak minus each run's OWN `t=0` is **74.465 GiB before and 74.300 GiB after**, with system `AnonPages` deltas of 38.012 vs 38.217 GiB and child `VmRSS` at each peak sample of 41.952 vs 42.090 GiB — three instruments inside ±0.25 GiB, both peaks sampled inside a ~1900% CPU stretch rather than in different phases. On a box as clean as the first run's the second binary's own demand would have left **40.59 GiB** available, above both the 12 GiB hard floor and the 8 GiB projection floor. **Two fixes, neither in this repository:** record `used_gib` at `t=0` in `PROVENANCE` and make every cross-run claim on the delta, on `rss_gib`, or refuse it; and find what held 26.8 GiB on `dgx:gpu0` before the job started — a leaked container or orphaned process from a previous job is the candidate, and it is what actually aborted the render. NOT fixed in flow: `runguard.py` and fleet job hygiene are outside the tree, and the second half needs a look at the box rather than a diff. Owed under `## Owed` of [`ltx25-text-linear-mem.md`](../../specs/ltx25-text-linear-mem.md) | bug |

## Resolution

-
