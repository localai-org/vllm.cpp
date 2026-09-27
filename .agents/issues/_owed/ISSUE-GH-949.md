ID: ISSUE-GH-949
Title: Nothing in the tree refuses a borrowed `vt::Tensor` that outlives the object owning its storage, and the ONLY instrument that catches one is `sanitize-cpu`, which is `continue-on-error` — that is how [#904](https://github.com/mudler/vllm.cpp/issues/904) landed. Measured in the #936 review rather than argued: with the #904 fix reverted, a plain Release build with no sanitizer runs the case 18/18 passed, 546 assertions, `rc=0`, because `dtype` lives in the `vt::Tensor` struct and not in the freed buffer, so no ordinary gate can see the dangling read. Three remedies are open and none is foregone: promote the lane once it has a `main` baseline, add a test that fails without a sanitizer, or reject the pattern statically — a prototype detector for a member access chained onto a call returning an owning type by value swept 1777 files with no hit but the defect. Anchors: the owning deleter `src/vllm/model_executor/models/ltx2_device.cpp:1088 @ 800dd082f`, the read `src/vt/cpu/cpu_layernorm.cpp:33 @ 800dd082f`. Listed under `## Owed` in [`ltx2-device-staged-view-uaf.md`](../../specs/ltx2-device-staged-view-uaf.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 949
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:261`

### Frozen archive evidence

> | [#949](https://github.com/mudler/vllm.cpp/issues/949) | — | Nothing in the tree refuses a borrowed `vt::Tensor` that outlives the object owning its storage, and the ONLY instrument that catches one is `sanitize-cpu`, which is `continue-on-error` — that is how [#904](https://github.com/mudler/vllm.cpp/issues/904) landed. Measured in the #936 review rather than argued: with the #904 fix reverted, a plain Release build with no sanitizer runs the case 18/18 passed, 546 assertions, `rc=0`, because `dtype` lives in the `vt::Tensor` struct and not in the freed buffer, so no ordinary gate can see the dangling read. Three remedies are open and none is foregone: promote the lane once it has a `main` baseline, add a test that fails without a sanitizer, or reject the pattern statically — a prototype detector for a member access chained onto a call returning an owning type by value swept 1777 files with no hit but the defect. Anchors: the owning deleter `src/vllm/model_executor/models/ltx2_device.cpp:1088 @ 800dd082f`, the read `src/vt/cpu/cpu_layernorm.cpp:33 @ 800dd082f`. Listed under `## Owed` in [`ltx2-device-staged-view-uaf.md`](../../specs/ltx2-device-staged-view-uaf.md) | bug |

## Resolution

-
