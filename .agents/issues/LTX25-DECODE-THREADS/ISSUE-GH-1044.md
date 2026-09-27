ID: ISSUE-GH-1044
Title: The three parallel dispatch sites [#1009](https://github.com/mudler/vllm.cpp/issues/1009) added to the LTX-2.5 conv video VAE share ONE work-stealing cursor, so reverting any SINGLE one of them is detected by nothing: `CausalConv3d`'s padding gather (`src/vllm/model_executor/models/ltx2_video_vae.cpp:170 @ 249418305`), its output nest (`:218`) and `Linear3d` (`:276`). The instrument, the case "the decode DISPATCHES its convolutions to the CPU threadpool", reads `Threadpool::ChunkAdd(0)` and that cursor is seeded once per pool (`src/vt/cpu/cpu_threadpool.cpp:438 @ 249418305`, advanced at `:455`), so it gates "at least ONE of the three dispatches", never each site. MEASURED as T1/T2/T3 in [`ltx25-decode-threads.md`](../../specs/ltx25-decode-threads.md) §8.6 and reproduced independently by that row's reviewer: each single-site revert BUILT with 0 errors and left ctest at exit 0 and 42/42 + 10/10 green; only reverting all three (T0) goes red. NOT a correctness hole -- 34 golden margins were byte-identical, the bit-identity case `memcmp`s five worker counts, and ThreadSanitizer is clean against an 84-race positive control -- but a site can silently go serial again and only a wall-clock nobody runs in CI would notice. Closing it needs a per-dispatch `Threadpool::RunCount()` bumped in `Run()` (`src/vt/cpu/cpu_threadpool.h:112 @ 249418305`; no such counter exists) and an EXACT expected count rather than `> 0`, plus a fixture carrying a `res_x_y` block, because `MakeLtx2ThreadFixture` sets `decoder_blocks = {}` and `Linear3d` is unreachable without one. A new gate needs its own red-before evidence and its own fresh review, so it is a row rather than an in-flow repair. Listed under `## 7. Owed` in [`ltx25-decode-threads.md`](../../specs/ltx25-decode-threads.md)
Row: LTX25-DECODE-THREADS
State: UNKNOWN
Kind: verification
GitHub: 1044
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:299`

### Frozen archive evidence

> | [#1044](https://github.com/mudler/vllm.cpp/issues/1044) | `LTX25-DECODE-THREADS` | The three parallel dispatch sites [#1009](https://github.com/mudler/vllm.cpp/issues/1009) added to the LTX-2.5 conv video VAE share ONE work-stealing cursor, so reverting any SINGLE one of them is detected by nothing: `CausalConv3d`'s padding gather (`src/vllm/model_executor/models/ltx2_video_vae.cpp:170 @ 249418305`), its output nest (`:218`) and `Linear3d` (`:276`). The instrument, the case "the decode DISPATCHES its convolutions to the CPU threadpool", reads `Threadpool::ChunkAdd(0)` and that cursor is seeded once per pool (`src/vt/cpu/cpu_threadpool.cpp:438 @ 249418305`, advanced at `:455`), so it gates "at least ONE of the three dispatches", never each site. MEASURED as T1/T2/T3 in [`ltx25-decode-threads.md`](../../specs/ltx25-decode-threads.md) §8.6 and reproduced independently by that row's reviewer: each single-site revert BUILT with 0 errors and left ctest at exit 0 and 42/42 + 10/10 green; only reverting all three (T0) goes red. NOT a correctness hole -- 34 golden margins were byte-identical, the bit-identity case `memcmp`s five worker counts, and ThreadSanitizer is clean against an 84-race positive control -- but a site can silently go serial again and only a wall-clock nobody runs in CI would notice. Closing it needs a per-dispatch `Threadpool::RunCount()` bumped in `Run()` (`src/vt/cpu/cpu_threadpool.h:112 @ 249418305`; no such counter exists) and an EXACT expected count rather than `> 0`, plus a fixture carrying a `res_x_y` block, because `MakeLtx2ThreadFixture` sets `decoder_blocks = {}` and `Linear3d` is unreachable without one. A new gate needs its own red-before evidence and its own fresh review, so it is a row rather than an in-flow repair. Listed under `## 7. Owed` in [`ltx25-decode-threads.md`](../../specs/ltx25-decode-threads.md) | verification |

## Resolution

-
