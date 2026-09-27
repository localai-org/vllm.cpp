ID: ISSUE-GH-1040
Title: `LTX25-DECODE-SPEED` ([#1006](https://github.com/mudler/vllm.cpp/issues/1006)) records three sets of PASSING measurements and **none has a retrievable evidence artifact**, which `.agents/verification.md` requires alongside the SHA, command, environment and exit status. Rung 1's 248 per-PID sampler rows, rung 2's 1082 rows and 347 side-car per-PID samples, rung 2's `run.log`, and §1.4's `~/work/ltx25-e2e/render8-console.log` all live only on `dgx.casa` — `ping -c 2 -W 3 dgx.casa` exits 1 with 100% packet loss and `Destination Host Unreachable` (checked 2026-08-16), this box's documented unified-memory OOM-reboot mode, which needs a physical power cycle. The spec's `REMOTE_UNVERIFIED` mark was correctly scoped to rung 2's EXIT REASON; this issue extends the same honesty to the passing numbers. **Second and more corrosive: neither rung's sampler cadence closes.** Rung 1 states 248 samples at 2 s over a 701 s run split 192+56 across 450 s and 164 s windows — at 2 s those windows hold 225 and 82, and 248 samples cover 496 s of 701. A draft of rung 2's §5 stated 347 per-PID samples at 2 s "over the first 1192 s" — at 2 s that window holds 596. No dropped-sample rate is recorded anywhere, so the wall each sample set covers is NOT derivable, and the spec now states sample counts and fractions rather than minute figures. The raw CSVs settle both in one pass. Filed while repairing the fresh review of [PR #1038](https://github.com/mudler/vllm.cpp/pull/1038); not fixable in that flow, because no edit to the tree produces a file on a host that does not answer. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1040
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:295`

### Frozen archive evidence

> | [#1040](https://github.com/mudler/vllm.cpp/issues/1040) | — | `LTX25-DECODE-SPEED` ([#1006](https://github.com/mudler/vllm.cpp/issues/1006)) records three sets of PASSING measurements and **none has a retrievable evidence artifact**, which `.agents/verification.md` requires alongside the SHA, command, environment and exit status. Rung 1's 248 per-PID sampler rows, rung 2's 1082 rows and 347 side-car per-PID samples, rung 2's `run.log`, and §1.4's `~/work/ltx25-e2e/render8-console.log` all live only on `dgx.casa` — `ping -c 2 -W 3 dgx.casa` exits 1 with 100% packet loss and `Destination Host Unreachable` (checked 2026-08-16), this box's documented unified-memory OOM-reboot mode, which needs a physical power cycle. The spec's `REMOTE_UNVERIFIED` mark was correctly scoped to rung 2's EXIT REASON; this issue extends the same honesty to the passing numbers. **Second and more corrosive: neither rung's sampler cadence closes.** Rung 1 states 248 samples at 2 s over a 701 s run split 192+56 across 450 s and 164 s windows — at 2 s those windows hold 225 and 82, and 248 samples cover 496 s of 701. A draft of rung 2's §5 stated 347 per-PID samples at 2 s "over the first 1192 s" — at 2 s that window holds 596. No dropped-sample rate is recorded anywhere, so the wall each sample set covers is NOT derivable, and the spec now states sample counts and fractions rather than minute figures. The raw CSVs settle both in one pass. Filed while repairing the fresh review of [PR #1038](https://github.com/mudler/vllm.cpp/pull/1038); not fixable in that flow, because no edit to the tree produces a file on a host that does not answer. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | bug |

## Resolution

-
