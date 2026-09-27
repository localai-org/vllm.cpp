ID: ISSUE-GH-1720
Title: **`WriteJson` reads `Elapsed()` and `Records()` under TWO separate acquisitions of the process-wide mutex, so `wall_seconds` and the record set are no longer one snapshot.** [PR #1711](https://github.com/mudler/vllm.cpp/pull/1711) moves `PhaseLog::WriteJson`'s clock read ABOVE its `ByStart(Records())` so the writer's own copy and sort stop being charged to the render's wall and therefore to `unaccounted_seconds` -- the defect [#1569](https://github.com/mudler/vllm.cpp/issues/1569) tracks -- and this issue is the cost of that repair recorded rather than hidden. Under the previous order the pair was effectively one snapshot in the direction that matters, because the clock was read LAST and `wall >= max(end_seconds)` held by construction; it no longer does. The observable if it broke is a NEGATIVE tail gap, which `gaps` reports and which `ltx2 phase log: the emitted table DECOMPOSES its residue into the gaps between leaves` refuses at `CHECK(seconds >= 0.0)`, so the table would say so rather than pass quietly. It is UNREACHABLE on the shipped path, and that is a property of the CALL SITE rather than of the function: both `WritePhaseLog` calls in `src/vllm/multimodal/ltx2_video.cpp` run after `generate_span.Close()`, that span is the last live scope, and `PhaseLog::Close` stops and JOINS the sampler before it returns when nothing is left live, so no thread can close a scope between those two lines on any path this project ships. A fresh review also failed to stage the inversion adversarially: 27,471 probes of a churn thread against a replica of the two statements produced zero. NOT FIXED IN FLOW: the real repair is to make the pair a single locked snapshot, and `Elapsed()` and `Records()` are separate public entry points on `PhaseLog`, so a combined one is a public API change and owes its own row, spec and red-first evidence rather than being smuggled into a repair commit for a different defect. Owned by [`ltx25-phase-instrument.md`](../../specs/ltx25-phase-instrument.md) `## Owed`
Row: LTX25-PHASE-INSTRUMENT
State: UNKNOWN
Kind: bug
GitHub: 1720
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:636`

### Frozen archive evidence

> | [#1720](https://github.com/mudler/vllm.cpp/issues/1720) | `LTX25-PHASE-INSTRUMENT` | **`WriteJson` reads `Elapsed()` and `Records()` under TWO separate acquisitions of the process-wide mutex, so `wall_seconds` and the record set are no longer one snapshot.** [PR #1711](https://github.com/mudler/vllm.cpp/pull/1711) moves `PhaseLog::WriteJson`'s clock read ABOVE its `ByStart(Records())` so the writer's own copy and sort stop being charged to the render's wall and therefore to `unaccounted_seconds` -- the defect [#1569](https://github.com/mudler/vllm.cpp/issues/1569) tracks -- and this issue is the cost of that repair recorded rather than hidden. Under the previous order the pair was effectively one snapshot in the direction that matters, because the clock was read LAST and `wall >= max(end_seconds)` held by construction; it no longer does. The observable if it broke is a NEGATIVE tail gap, which `gaps` reports and which `ltx2 phase log: the emitted table DECOMPOSES its residue into the gaps between leaves` refuses at `CHECK(seconds >= 0.0)`, so the table would say so rather than pass quietly. It is UNREACHABLE on the shipped path, and that is a property of the CALL SITE rather than of the function: both `WritePhaseLog` calls in `src/vllm/multimodal/ltx2_video.cpp` run after `generate_span.Close()`, that span is the last live scope, and `PhaseLog::Close` stops and JOINS the sampler before it returns when nothing is left live, so no thread can close a scope between those two lines on any path this project ships. A fresh review also failed to stage the inversion adversarially: 27,471 probes of a churn thread against a replica of the two statements produced zero. NOT FIXED IN FLOW: the real repair is to make the pair a single locked snapshot, and `Elapsed()` and `Records()` are separate public entry points on `PhaseLog`, so a combined one is a public API change and owes its own row, spec and red-first evidence rather than being smuggled into a repair commit for a different defect. Owned by [`ltx25-phase-instrument.md`](../../specs/ltx25-phase-instrument.md) `## Owed` | bug |

## Resolution

-
