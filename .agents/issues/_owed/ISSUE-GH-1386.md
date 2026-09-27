ID: ISSUE-GH-1386
Title: `tools/bench/gpu_clock_state.py`'s `QUERY_FIELDS` collects nine fields and **none of them is thermal and none is electrical**, so the driver's own `SwThermalSlowdown` label can never be checked against a die reading on any window this helper has ever recorded. The measured consequence is that the nine windows of 2026-08-19 cannot distinguish a load transition from a thermal excursion. The concrete evidence is `clock-c1-r1.samples.json` in `/mnt/nas_share/rc/q38bf16/out/bench-20260819T035148Z/`: ours c1 r1 dips five times on the same period at the same `utilization.gpu = 96` — 48.83 s / 2177 MHz, 80.60 s / 2320 MHz, 109.28 s / 2210 MHz, 137.98 s / 2359 MHz, 166.07 s / 2268 MHz — and **two of those five carry `0x0000000000000000`**, no throttle bit at all (2210 and 2359), while three carry `0x20`. The 2210 MHz unlabelled dip is deeper than two of the three labelled ones, so the driver labels comparable excursions inconsistently and the bit alone cannot decide it. What would settle it: add `temperature.gpu` and `power.draw` to `QUERY_FIELDS`. That changes the clock-record schema, so it owes its own row and spec. Split out of [#1354](https://github.com/mudler/vllm.cpp/issues/1354) and owed under `## Owed` in [lease-clock-pinning.md](../../specs/lease-clock-pinning.md)
Row: -
State: UNKNOWN
Kind: gap
GitHub: 1386
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:476`

### Frozen archive evidence

> | [#1386](https://github.com/mudler/vllm.cpp/issues/1386) | — | `tools/bench/gpu_clock_state.py`'s `QUERY_FIELDS` collects nine fields and **none of them is thermal and none is electrical**, so the driver's own `SwThermalSlowdown` label can never be checked against a die reading on any window this helper has ever recorded. The measured consequence is that the nine windows of 2026-08-19 cannot distinguish a load transition from a thermal excursion. The concrete evidence is `clock-c1-r1.samples.json` in `/mnt/nas_share/rc/q38bf16/out/bench-20260819T035148Z/`: ours c1 r1 dips five times on the same period at the same `utilization.gpu = 96` — 48.83 s / 2177 MHz, 80.60 s / 2320 MHz, 109.28 s / 2210 MHz, 137.98 s / 2359 MHz, 166.07 s / 2268 MHz — and **two of those five carry `0x0000000000000000`**, no throttle bit at all (2210 and 2359), while three carry `0x20`. The 2210 MHz unlabelled dip is deeper than two of the three labelled ones, so the driver labels comparable excursions inconsistently and the bit alone cannot decide it. What would settle it: add `temperature.gpu` and `power.draw` to `QUERY_FIELDS`. That changes the clock-record schema, so it owes its own row and spec. Split out of [#1354](https://github.com/mudler/vllm.cpp/issues/1354) and owed under `## Owed` in [lease-clock-pinning.md](../../specs/lease-clock-pinning.md) | gap |

## Resolution

-
