ID: ISSUE-GH-1906
Title: **`test_ltx2_video`'s `/tmp` workspace is deleted by another process while a run is using it, and the abort costs 82 assertions with `failures="0"`.** The fixture writes every render into `"/tmp/vllm_ltx2_video_" + getpid() + "_" + counter` (`test_ltx2_video.cpp:67`), which is prefix-predictable and outside any per-run private directory. One run of seven at `ced0ab639` aborted with `cannot write /tmp/vllm_ltx2_video_970765_3/multichunk/audio.wav` after `create_directories` had succeeded and the frames had already been written to that same directory; it is not ENOSPC (49 GB free, and an ENOSPC open succeeds so the failure would read `short write`) and not a descriptor limit (`ulimit -n` 1048576, `file-nr` 6120). A sentinel planted at `/tmp/vllm_ltx2_video_999999_0` was GONE within two minutes while a second planted four minutes later survived twenty, so the deletion is episodic; `systemd-tmpfiles-clean` last fired 19 hours earlier, `grep -rn vllm_ltx2_video` over the tree returns only the fixture's own line, and a second agent was running its own build of this suite from another worktree throughout. An exception aborts a doctest case where it stands, so 89 of that case's 171 assertions ran and the run's total was 82 short with no failure recorded -- silent coverage loss wearing a normal number, and the second way this suite's total moves after [#1885](https://github.com/mudler/vllm.cpp/issues/1885). NOT fixed in flow: a random suffix under the same prefix still dies to `rm -rf /tmp/vllm_ltx2_video_*`, so where the fixture writes is a decision. Listed under `## Owed` in [ltx25-test-determinism.md](../../specs/ltx25-test-determinism.md)
Row: LTX25-TEST-DETERMINISM
State: UNKNOWN
Kind: bug
GitHub: 1906
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:718`

### Frozen archive evidence

> | [#1906](https://github.com/mudler/vllm.cpp/issues/1906) | `LTX25-TEST-DETERMINISM` | **`test_ltx2_video`'s `/tmp` workspace is deleted by another process while a run is using it, and the abort costs 82 assertions with `failures="0"`.** The fixture writes every render into `"/tmp/vllm_ltx2_video_" + getpid() + "_" + counter` (`test_ltx2_video.cpp:67`), which is prefix-predictable and outside any per-run private directory. One run of seven at `ced0ab639` aborted with `cannot write /tmp/vllm_ltx2_video_970765_3/multichunk/audio.wav` after `create_directories` had succeeded and the frames had already been written to that same directory; it is not ENOSPC (49 GB free, and an ENOSPC open succeeds so the failure would read `short write`) and not a descriptor limit (`ulimit -n` 1048576, `file-nr` 6120). A sentinel planted at `/tmp/vllm_ltx2_video_999999_0` was GONE within two minutes while a second planted four minutes later survived twenty, so the deletion is episodic; `systemd-tmpfiles-clean` last fired 19 hours earlier, `grep -rn vllm_ltx2_video` over the tree returns only the fixture's own line, and a second agent was running its own build of this suite from another worktree throughout. An exception aborts a doctest case where it stands, so 89 of that case's 171 assertions ran and the run's total was 82 short with no failure recorded -- silent coverage loss wearing a normal number, and the second way this suite's total moves after [#1885](https://github.com/mudler/vllm.cpp/issues/1885). NOT fixed in flow: a random suffix under the same prefix still dies to `rm -rf /tmp/vllm_ltx2_video_*`, so where the fixture writes is a decision. Listed under `## Owed` in [ltx25-test-determinism.md](../../specs/ltx25-test-determinism.md) | bug |

## Resolution

-
