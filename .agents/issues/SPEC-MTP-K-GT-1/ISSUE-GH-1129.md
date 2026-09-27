ID: ISSUE-GH-1129
Title: The pinned vLLM oracle has NO lease-compliant path on `dgx.casa`, confirmed from two independent directions. The HOST has carried no toolchain since the 2026-08-14 reimage (no `gcc`, `cc`, `clang`, `ninja` or `nvcc`, no `/usr/include/stdio.h`), so the recorded cure is `sudo -n docker run` against `vllmcpp-build:gb10` or `nvidia/cuda:13.0.1-devel-ubuntu24.04`, reached over `ssh`, which bypasses the `rc` lease and makes the fleet report `dgx:gpu0` free while somebody is on it. `994d30b5d` (PR [#1116](https://github.com/mudler/vllm.cpp/pull/1116)) recorded that half and left the container re-check owed. The re-check is now DONE and CONFIRMS the blocker: probed through `rc run -d dgx:gpu0` on 2026-08-17 (job `ff28ada1-0cd3-4867-bf9b-f67050d0608b`), the leased worker runs as user `rc` in a k3s pod and carries no `gcc`, `cc`, `clang`, `nvcc`, `ninja`, `cmake`, `make`, `python3`, `pip`, `docker`, `sudo`, `git`, `ssh` or `curl`, no `/usr/include/stdio.h` and no `/usr/local/cuda*`, and `/home/mudler` does not exist inside it, so it cannot reach `~/venvs/vllm-oracle-pin-555967922` and could not start it if it could. `rc run` has no `--image` flag. So NO vLLM leg of any row can currently run on `dgx.casa` by any lease-compliant path, which blocks every oracle-dependent row: [#1003](https://github.com/mudler/vllm.cpp/issues/1003)'s twelve re-takes, [#915](https://github.com/mudler/vllm.cpp/issues/915)'s withheld cells, [#821](https://github.com/mudler/vllm.cpp/issues/821), and the MTP adjudication plus vLLM leg owed by [#81](https://github.com/mudler/vllm.cpp/issues/81). The only recorded lever is `/workspace`, which is NAS-backed, writable from the worker and the same folder on `dgx` and `thor`. The migration is deliberately NOT designed there. Listed under `## Owed` in [`mtp-k-gt-1.md`](../../specs/mtp-k-gt-1.md)
Row: SPEC-MTP-K-GT-1
State: UNKNOWN
Kind: bug
GitHub: 1129
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:331`

### Frozen archive evidence

> | [#1129](https://github.com/mudler/vllm.cpp/issues/1129) | `SPEC-MTP-K-GT-1` | The pinned vLLM oracle has NO lease-compliant path on `dgx.casa`, confirmed from two independent directions. The HOST has carried no toolchain since the 2026-08-14 reimage (no `gcc`, `cc`, `clang`, `ninja` or `nvcc`, no `/usr/include/stdio.h`), so the recorded cure is `sudo -n docker run` against `vllmcpp-build:gb10` or `nvidia/cuda:13.0.1-devel-ubuntu24.04`, reached over `ssh`, which bypasses the `rc` lease and makes the fleet report `dgx:gpu0` free while somebody is on it. `994d30b5d` (PR [#1116](https://github.com/mudler/vllm.cpp/pull/1116)) recorded that half and left the container re-check owed. The re-check is now DONE and CONFIRMS the blocker: probed through `rc run -d dgx:gpu0` on 2026-08-17 (job `ff28ada1-0cd3-4867-bf9b-f67050d0608b`), the leased worker runs as user `rc` in a k3s pod and carries no `gcc`, `cc`, `clang`, `nvcc`, `ninja`, `cmake`, `make`, `python3`, `pip`, `docker`, `sudo`, `git`, `ssh` or `curl`, no `/usr/include/stdio.h` and no `/usr/local/cuda*`, and `/home/mudler` does not exist inside it, so it cannot reach `~/venvs/vllm-oracle-pin-555967922` and could not start it if it could. `rc run` has no `--image` flag. So NO vLLM leg of any row can currently run on `dgx.casa` by any lease-compliant path, which blocks every oracle-dependent row: [#1003](https://github.com/mudler/vllm.cpp/issues/1003)'s twelve re-takes, [#915](https://github.com/mudler/vllm.cpp/issues/915)'s withheld cells, [#821](https://github.com/mudler/vllm.cpp/issues/821), and the MTP adjudication plus vLLM leg owed by [#81](https://github.com/mudler/vllm.cpp/issues/81). The only recorded lever is `/workspace`, which is NAS-backed, writable from the worker and the same folder on `dgx` and `thor`. The migration is deliberately NOT designed there. Listed under `## Owed` in [`mtp-k-gt-1.md`](../../specs/mtp-k-gt-1.md) | bug |

## Resolution

-
