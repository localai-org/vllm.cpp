# SPEC — `LLAMA-CPP-PIN-ADVANCE` — advance the stock llama-cpp oracle pin past the Kolibri-1 support merge, reconciling every affected gate

- Owning row: `LLAMA-CPP-PIN-ADVANCE`
- Kind: oracle/pin maintenance (upstream sync with full reconciliation)
- Issue: created in the same flow (row-scoped local issue)

## Why now

The `llama-cpp` oracle pins stock release `b10451`, which predates
Kolibri-1 support (PR #30673, merged 2026-10-17) and the router-precision
removal (#30865, 2026-10-20 — numerics-relevant). The KOLIBRI1-GGUF-ARM
row (landed lineage, PR #3527) named the pin advance as its follow-up so a
**quant-matched floor comparison** becomes possible: llama.cpp can serve
the mudler APEX GGUFs directly (standard ggml types), giving the kolibri1
GGUF arm its first external denominator.

Per AGENTS.md upstream-sync: advance the pin ONLY after reconciling every
affected row and gate, and the advanced pin must demonstrably build and
run the models the floors are taken on.

## The advance

- Identify the stock release tag that first carries #30673 + #30865
  (the first release tag after 2026-10-20 — verify by tag listing +
  the merge commit's presence in the tag's history, recorded in the
  evidence).
- Update `.agents/oracles/llama-cpp.md` pin block (+ pin_label, pinned_on,
  gateable evidence per the README contract) and the
  ORACLE-PIN-SNAPSHOT/parity surfaces that restate it.
- Build the new pin on this host (aarch64) and run the gateability
  legs: kolibri-1 APEX (the GGUF this row exists for) + one pre-existing
  floor model (the qwen38 ladder's artifact class) to prove no regression
  in the floors the pin already supplies.

## Reconciliation inventory (every consumer of the old pin)

Grep the tree for `b10451` and the llama-cpp floor suites: known consumers
include the x86 llamacpp floor suite (`test_cpu_x86_llamacpp_floor`), the
qwen38 ladder gates, and any floor numbers recorded against b10451. Each
consumer either (a) re-measures on the new pin (record the delta), or
(b) is explicitly re-anchored to a frozen artifact with a dated note if
re-measurement is impossible — the qwen4exp/glm5next pattern.

## What this row is NOT

- NOT the kolibri1 quant-matched comparison itself: this row delivers the
  advanced, gateable pin + the reconciliation; the comparison measurement
  (llama.cpp vs our GGUF arm on the same APEX file) is the row's final
  recorded number if it fits, else the named next step.
- NOT a flint-side change: the llama-cpp-flint record (PR head 1b44e3077)
  is separate by design and stays pinned as is.

## Gates

check-oracle-pins (+ self-test), the reconciliation consumers above green
on the new pin, check-agent-record, style, trailers, preflight (named
host dotnet failure). Device-free row.

## Red-first / evidence

The pin advance is upstream sync, not a semantic checker change: the
red-first is the reconciliation inventory (every b10451 consumer named
BEFORE the bump, each disposition recorded) + the gateability legs. All
of it lands in docs/bench-evidence/llama-cpp-pin-advance-20261031/.

## Stop conditions

- The first post-merge release cannot build on aarch64 → try the next
  tag; if none builds, STOP: the advance stalls on an external blocker,
  recorded as such.
- A reconciled consumer's numbers move beyond its own band on the new
  pin → that is a finding to record (the floor MOVED — that is what pin
  advances do); if a move invalidates a landed row's claim, STOP and
  adjudicate before landing.

## Outcome

(added 2026-10-31, row landed on `row/LLAMA-CPP-PIN-ADVANCE`)

- **Tag**: `b11765` (`9df0b73c707900b7277a46b5ae52ff7e7e567d13`, 2026-10-20) —
  the #30865 merge commit itself; the first release tag carrying both #30673
  and #30865 (verified through the compare API and a fresh clone; the repo's
  newer semver series starts later, `v0.8.0` 2026-10-28, and also carries both —
  recorded as the fallback).
- **Gateability**: release build (GGML_NATIVE=ON, aarch64) on this host serves
  the kolibri1 APEX I-Micro and the Qwen3.8-27B Q4_K_M floor artifact; the
  flint artifact refuses at the stock pin, correctly (separate PR-head oracle).
- **Pin**: `llama-cpp` advanced `62318cd57`/`b11709` → `9df0b73c`/`b11765`,
  `pinned_on = 2026-10-31`, `gateable = yes`. The scoped `llama-cpp-kolibri1`
  record already pinned `b11765` and keeps its own file with a dated note.
- **Comparison (the payoff)**: on the same `Kolibri-1-APEX-I-Micro.gguf`, bench
  zone and thread counts, llama.cpp measures 17.98 ± 0.12 tok/s (16t) and
  20.18 ± 0.61 tok/s (32t) against our arm's 4.97 / 6.55 — the floor is
  3.62x / 3.08x ahead. The gap is recorded, not adjudicated.
- **Rejected alternative**: advancing to `v0.8.0` instead of `b11765` — the
  spec names the FIRST release tag after the merges, and `b11765` is it;
  `v0.8.0` would add eight days of unrelated commits to the floor without
  answering the row's question any better.
- No floor moved: no recorded llama.cpp number in the tree was measured at the
  retired live pin, so neither stop condition fired.

## Git integration

One pull request, `row/LLAMA-CPP-PIN-ADVANCE`, spec committed first.
