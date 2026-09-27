ID: ISSUE-GH-1080
Title: `scripts/check-doc-checkpoint.py:153` matches a spec's live-position section with `^##\s+Now\s*$`, and specs in this tree write it as `## N. Now`, so `spec_now_errors` reports "has no `## Now` section" about a section that is present and current. It fires only when a row moves lifecycle state. `nemotron-h-model.md` was one of them and is repaired in flow by [#1074](https://github.com/mudler/vllm.cpp/issues/1074), which is the change that made it the spec a moving row links. Re-measured at `b626be75a` after that repair: 15 specs still write the numbered spelling — `gate-audit-branch-evidence`, `ltx25-a2v-audio-input`, `ltx25-image-conditioning`, `ltx25-t2a-one-stage`, `ltx25-token-append`, `ltx2-device-staged-view-uaf`, `mamba2-ssd`, `nas-mount-path`, `nemotron-h-a2p-paged-forward`, `nemotron-h-a2q1-fp8-mamba`, `nemotron-h-a2q2b-realckpt-lmhead`, `nemotron-h-a2q2-nvfp4-moe-lmhead`, `nemotron-h-abi-e2e`, `offload-docs-refusal`, `registry-downcast-sweep`. The population GREW from the twelve this was filed against, which is the argument for the checker-semantics fix over a rename sweep. Either close needs its own spec and a red-before test, so neither rides in a records reconcile. `tests/scripts/` covers `NOW_SECTION` nowhere, which is how the mismatch survived. Listed under `## Owed` in [`nemotron-h-model.md`](../../specs/nemotron-h-model.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1080
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:481`

### Frozen archive evidence

> | [#1080](https://github.com/mudler/vllm.cpp/issues/1080) | — | `scripts/check-doc-checkpoint.py:153` matches a spec's live-position section with `^##\s+Now\s*$`, and specs in this tree write it as `## N. Now`, so `spec_now_errors` reports "has no `## Now` section" about a section that is present and current. It fires only when a row moves lifecycle state. `nemotron-h-model.md` was one of them and is repaired in flow by [#1074](https://github.com/mudler/vllm.cpp/issues/1074), which is the change that made it the spec a moving row links. Re-measured at `b626be75a` after that repair: 15 specs still write the numbered spelling — `gate-audit-branch-evidence`, `ltx25-a2v-audio-input`, `ltx25-image-conditioning`, `ltx25-t2a-one-stage`, `ltx25-token-append`, `ltx2-device-staged-view-uaf`, `mamba2-ssd`, `nas-mount-path`, `nemotron-h-a2p-paged-forward`, `nemotron-h-a2q1-fp8-mamba`, `nemotron-h-a2q2b-realckpt-lmhead`, `nemotron-h-a2q2-nvfp4-moe-lmhead`, `nemotron-h-abi-e2e`, `offload-docs-refusal`, `registry-downcast-sweep`. The population GREW from the twelve this was filed against, which is the argument for the checker-semantics fix over a rename sweep. Either close needs its own spec and a red-before test, so neither rides in a records reconcile. `tests/scripts/` covers `NOW_SECTION` nowhere, which is how the mismatch survived. Listed under `## Owed` in [`nemotron-h-model.md`](../../specs/nemotron-h-model.md) | bug |

## Resolution

-
