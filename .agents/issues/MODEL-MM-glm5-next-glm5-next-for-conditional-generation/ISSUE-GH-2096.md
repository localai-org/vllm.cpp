ID: ISSUE-GH-2096
Title: **GLM-5.3-Flash gates against `transformers` and its lane-scoped `v5.16.1` pin is unwritten: write it.** No oracle other than `transformers` implements `glm5_next` -- vLLM implements nothing at the parity pin `555967922` or at `main`, and vllm#53906 is OPEN and therefore inadmissible. `transformers` carries the architecture from `eb4d9e2a64a0` (transformers#48342, merged 2026-08-26T14:26:41Z) and the FIRST release carrying it is `v5.16.1`, bounded rather than assumed: `modeling_glm5_next.py` is HTTP 200 at `v5.16.1`, 404 at `v5.16.0` and 404 at `v5.15.1`, re-measured 2026-08-27. The registry pin is `5.14.1` and does not contain `Glm5Next`, so this row needs a lane-scoped second pin with `gateable = no`, expiring when vLLM registers `glm5_next`. Discharges O12 in [glm5-next-flash.md](../../specs/glm5-next-flash.md). W0 of campaign issue #1998
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: record
GitHub: 2096
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:809`

### Frozen archive evidence

> | [#2096](https://github.com/mudler/vllm.cpp/issues/2096) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **GLM-5.3-Flash gates against `transformers` and its lane-scoped `v5.16.1` pin is unwritten: write it.** No oracle other than `transformers` implements `glm5_next` -- vLLM implements nothing at the parity pin `555967922` or at `main`, and vllm#53906 is OPEN and therefore inadmissible. `transformers` carries the architecture from `eb4d9e2a64a0` (transformers#48342, merged 2026-08-26T14:26:41Z) and the FIRST release carrying it is `v5.16.1`, bounded rather than assumed: `modeling_glm5_next.py` is HTTP 200 at `v5.16.1`, 404 at `v5.16.0` and 404 at `v5.15.1`, re-measured 2026-08-27. The registry pin is `5.14.1` and does not contain `Glm5Next`, so this row needs a lane-scoped second pin with `gateable = no`, expiring when vLLM registers `glm5_next`. Discharges O12 in [glm5-next-flash.md](../../specs/glm5-next-flash.md). W0 of campaign issue #1998 | record |

## Resolution

-
