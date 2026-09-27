ID: ISSUE-GH-1014
Title: The ~59 GiB an LTX-2.5 render loses is **not in the decode**, established twice: MEASURED heap peak 361.72 MiB by exact `operator new` accounting ([`ltx25-tiled-decode.md`](../../specs/ltx25-tiled-decode.md) `## Outcome` item 2), and a COMPUTED 9.649 GiB ceiling assuming nothing is ever freed. Hypothesis, NOT closed: model residency — `docs/USAGE.md:862-864 @ 332aed738` already accounts for ~68 GiB staged and resident before any decode instruction, and the 320x192/49f render trace shows ~72 GiB acquired in the first ten minutes and held flat for two hours. A 2 s one-clock trace of `MemAvailable` + `VmRSS`/`Anonymous` + the CUDA compute-app footprint across the load/denoise/decode boundary settles it; if the fall is at the decode boundary instead, the next hypothesis is a CUDA or `mmap` mapping. Also owes the **`docs/USAGE.md:868 @ 332aed738`** "loses about 59 GB in 24 seconds inside the decode" correction — **`:868`, not `:873-874`**, which is a DIFFERENT sentence ("most of a 320x192/25f render is spent single-threaded in the host VAE decode at 0% GPU") owed to [#1024](https://github.com/mudler/vllm.cpp/issues/1024); a correction aimed at `:873-874` rewrites the wrong sentence. Lever 5. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1014
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:290`

### Frozen archive evidence

> | [#1014](https://github.com/mudler/vllm.cpp/issues/1014) | — | The ~59 GiB an LTX-2.5 render loses is **not in the decode**, established twice: MEASURED heap peak 361.72 MiB by exact `operator new` accounting ([`ltx25-tiled-decode.md`](../../specs/ltx25-tiled-decode.md) `## Outcome` item 2), and a COMPUTED 9.649 GiB ceiling assuming nothing is ever freed. Hypothesis, NOT closed: model residency — `docs/USAGE.md:862-864 @ 332aed738` already accounts for ~68 GiB staged and resident before any decode instruction, and the 320x192/49f render trace shows ~72 GiB acquired in the first ten minutes and held flat for two hours. A 2 s one-clock trace of `MemAvailable` + `VmRSS`/`Anonymous` + the CUDA compute-app footprint across the load/denoise/decode boundary settles it; if the fall is at the decode boundary instead, the next hypothesis is a CUDA or `mmap` mapping. Also owes the **`docs/USAGE.md:868 @ 332aed738`** "loses about 59 GB in 24 seconds inside the decode" correction — **`:868`, not `:873-874`**, which is a DIFFERENT sentence ("most of a 320x192/25f render is spent single-threaded in the host VAE decode at 0% GPU") owed to [#1024](https://github.com/mudler/vllm.cpp/issues/1024); a correction aimed at `:873-874` rewrites the wrong sentence. Lever 5. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | bug |

## Resolution

-
