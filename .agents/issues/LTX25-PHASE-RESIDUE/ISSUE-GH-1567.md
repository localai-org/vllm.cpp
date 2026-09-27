ID: ISSUE-GH-1567
Title: the res_2s sampler arm has no `denoise.update` anchor, so its denoise would be decomposed on one arm only. Filed by `LTX25-PHASE-RESIDUE` and deliberately NOT landed with it, because no gate in this tree renders on that arm and an anchor no gate runs is dead code. `Ltx2Res2sDenoisingLoop` runs its own post-process and step behind `Ltx2Res2sHooks`, so the anchor needs a hook rather than a statement. It lives in `ltx2_samplers.cpp`, is declared in `ltx2_samplers.h` beside the hooks struct, and is called from `ltx2_video.cpp`. **NOT `ltx2_res2s.cpp`**: #1556's spec named that file and it has never existed here, which `git log --all --diff-filter=A` confirms; #1567's forge text names no file at all, so the wrong anchor came from the spec rather than from the issue. Doubly owed while [#1668](https://github.com/mudler/vllm.cpp/issues/1668) is open, since the first-order arm has no anchor either. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md)
Row: LTX25-PHASE-RESIDUE
State: UNKNOWN
Kind: bug
GitHub: 1567
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:603`

### Frozen archive evidence

> | [#1567](https://github.com/mudler/vllm.cpp/issues/1567) | `LTX25-PHASE-RESIDUE` | the res_2s sampler arm has no `denoise.update` anchor, so its denoise would be decomposed on one arm only. Filed by `LTX25-PHASE-RESIDUE` and deliberately NOT landed with it, because no gate in this tree renders on that arm and an anchor no gate runs is dead code. `Ltx2Res2sDenoisingLoop` runs its own post-process and step behind `Ltx2Res2sHooks`, so the anchor needs a hook rather than a statement. It lives in `ltx2_samplers.cpp`, is declared in `ltx2_samplers.h` beside the hooks struct, and is called from `ltx2_video.cpp`. **NOT `ltx2_res2s.cpp`**: #1556's spec named that file and it has never existed here, which `git log --all --diff-filter=A` confirms; #1567's forge text names no file at all, so the wrong anchor came from the spec rather than from the issue. Doubly owed while [#1668](https://github.com/mudler/vllm.cpp/issues/1668) is open, since the first-order arm has no anchor either. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md) | bug |

## Resolution

-
