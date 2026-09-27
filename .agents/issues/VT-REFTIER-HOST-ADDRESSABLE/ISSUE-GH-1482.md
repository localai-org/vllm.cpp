ID: ISSUE-GH-1482
Title: **The remainder of [#844](https://github.com/mudler/vllm.cpp/issues/844), split out so it keeps an open tracker after the crash itself is fixed.** #844 asked for four things. The reference tier no longer hands a device tensor to a host kernel, and the banner no longer claims correctness it cannot hold; those two landed with this row. Two did not. Item 1 also wants the refusal to name the BUILD FEATURE that would have provided the native kernel: the landed message names the op, the device, and which tier precondition the device failed, but not `this build has no cutlass-fp8`, which is the sentence that would have turned #1435's measurement around in one read. It is absent because the op table holds no such mapping -- an `OpId` does not know which CMake feature cell compiles its registering translation unit -- and inferring one by scanning a kernel body for `cutlass` tokens was already argued and rejected in [`vt-fp8-quant-arch-gate.md`](../../specs/vt-fp8-quant-arch-gate.md) `## Outcome` as transitive through helpers, hence both false-positive and false-negative. A real fix needs a declared, checkable mapping, which is a design question and not a message change. Item 4 -- whether a CUDA build lacking CUTLASS should warn at ENGINE CONSTRUCTION rather than only at configure time -- is untouched. The refusal fires at first dispatch of the missing op, which can be deep inside a model forward, long after the configure log is gone. Filed rather than deferred silently: the fixing pull request carries `Fixes #844`, so without this row the remainder would point at a closed issue and have no owner, which is the ownerless-owed state the protocol forbids. Listed under `## Owed` in [`vt-reference-tier-host-addressable.md`](../../specs/vt-reference-tier-host-addressable.md)
Row: VT-REFTIER-HOST-ADDRESSABLE
State: UNKNOWN
Kind: bug
GitHub: 1482
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:518`

### Frozen archive evidence

> | [#1482](https://github.com/mudler/vllm.cpp/issues/1482) | `VT-REFTIER-HOST-ADDRESSABLE` | **The remainder of [#844](https://github.com/mudler/vllm.cpp/issues/844), split out so it keeps an open tracker after the crash itself is fixed.** #844 asked for four things. The reference tier no longer hands a device tensor to a host kernel, and the banner no longer claims correctness it cannot hold; those two landed with this row. Two did not. Item 1 also wants the refusal to name the BUILD FEATURE that would have provided the native kernel: the landed message names the op, the device, and which tier precondition the device failed, but not `this build has no cutlass-fp8`, which is the sentence that would have turned #1435's measurement around in one read. It is absent because the op table holds no such mapping -- an `OpId` does not know which CMake feature cell compiles its registering translation unit -- and inferring one by scanning a kernel body for `cutlass` tokens was already argued and rejected in [`vt-fp8-quant-arch-gate.md`](../../specs/vt-fp8-quant-arch-gate.md) `## Outcome` as transitive through helpers, hence both false-positive and false-negative. A real fix needs a declared, checkable mapping, which is a design question and not a message change. Item 4 -- whether a CUDA build lacking CUTLASS should warn at ENGINE CONSTRUCTION rather than only at configure time -- is untouched. The refusal fires at first dispatch of the missing op, which can be deep inside a model forward, long after the configure log is gone. Filed rather than deferred silently: the fixing pull request carries `Fixes #844`, so without this row the remainder would point at a closed issue and have no owner, which is the ownerless-owed state the protocol forbids. Listed under `## Owed` in [`vt-reference-tier-host-addressable.md`](../../specs/vt-reference-tier-host-addressable.md) | bug |

## Resolution

-
