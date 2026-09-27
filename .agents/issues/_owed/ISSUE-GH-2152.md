ID: ISSUE-GH-2152
Title: **The c=8 ladder rung has a 127% spread, so every single-run comparison at that rung is ungated, including the standing vLLM and SGLang positions.** Seven interleaved runs on 2026-08-28, two builds, one lease, one hour, every arm re-measured: `16ebcac4b` read 56.22 / 51.29 / 36.82 out tok/s and `5e9d81dad` read 34.66 / 35.49 / 43.30 / 78.86. One UNCHANGED binary spans 52%, the other 127%. The instrument's spread is larger than every effect it has been asked to detect. The 5.9% c=8 figure quoted throughout this repository comes from a 4-run study that sampled a stable window and has since been used as though it bounded the rung; it does not, and a number quoted often became treated as measured. Everything gated at c=8 with n=1 per arm is therefore ungated: #2148's 38% (void, see [#2151](https://github.com/mudler/vllm.cpp/issues/2151)), the W12 and W13 c=8 attributions, and the "parity with vLLM, 23% behind SGLang" position. Owes three things — a repeat count DERIVED from the measured spread rather than assumed, interleaved arms plus a terminal control in the harness itself so a drifting box invalidates its own run instead of returning a confident number, and a cause for the drift (a clock pin outliving a lease is the first hypothesis and is untested). Owed under `## Owed` in [specs/reorder-threshold-wiring.md](../../specs/reorder-threshold-wiring.md) until a row picks it up
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2152
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:835`

### Frozen archive evidence

> | [#2152](https://github.com/mudler/vllm.cpp/issues/2152) | — | **The c=8 ladder rung has a 127% spread, so every single-run comparison at that rung is ungated, including the standing vLLM and SGLang positions.** Seven interleaved runs on 2026-08-28, two builds, one lease, one hour, every arm re-measured: `16ebcac4b` read 56.22 / 51.29 / 36.82 out tok/s and `5e9d81dad` read 34.66 / 35.49 / 43.30 / 78.86. One UNCHANGED binary spans 52%, the other 127%. The instrument's spread is larger than every effect it has been asked to detect. The 5.9% c=8 figure quoted throughout this repository comes from a 4-run study that sampled a stable window and has since been used as though it bounded the rung; it does not, and a number quoted often became treated as measured. Everything gated at c=8 with n=1 per arm is therefore ungated: #2148's 38% (void, see [#2151](https://github.com/mudler/vllm.cpp/issues/2151)), the W12 and W13 c=8 attributions, and the "parity with vLLM, 23% behind SGLang" position. Owes three things — a repeat count DERIVED from the measured spread rather than assumed, interleaved arms plus a terminal control in the harness itself so a drifting box invalidates its own run instead of returning a confident number, and a cause for the drift (a clock pin outliving a lease is the first hypothesis and is untested). Owed under `## Owed` in [specs/reorder-threshold-wiring.md](../../specs/reorder-threshold-wiring.md) until a row picks it up | bug |

## Resolution

-
