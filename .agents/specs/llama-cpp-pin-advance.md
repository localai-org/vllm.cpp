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

## Git integration

One pull request, `row/LLAMA-CPP-PIN-ADVANCE`, spec committed first.
