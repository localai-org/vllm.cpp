# SPEC — `FLINT-ROUTER-TOTALORDER` — the tie-break that makes the router selection deterministic, then the parallel top-k it unlocks

- Owning row: `FLINT-ROUTER-TOTALORDER`
- Kind: bug fix (determinism) + performance (the unlocked parallel top-k)
- Issue: created in the same flow (row-scoped local issue; supersedes the
  FLINT-CPU-DESCENT-2 router-top-k STOP)

## The problem, precisely

The Flint router selects top-6-of-384 via (score, index) pairs and
`std::partial_sort` with a comparator on score only
(`src/vllm/model_executor/models/flint.cpp`, the router block). Among
exact f32 ties the order is UNSPECIFIED — the standard does not define
which of equal elements a partial_sort produces, so the selected expert
set is implementation-defined on tie inputs. The router feeds the
gather, so a different expert set changes tokens.

FLINT-CPU-DESCENT-2 stopped honestly: a parallel/radix replacement
cannot be proven bit-identical to an unspecified order. But the stop
conflated "the current order is unspecified" with "no deterministic
order exists" — the pairs already carry the index, and one tie-break
makes the comparator a TOTAL order.

## The fix

The comparator becomes:

```cpp
return a.first != b.first ? a.first > b.first
                          : a.second < b.second;  // lowest index wins ties
```

Lowest-index-wins is the stable-sort semantics — the conventional
choice, and it matches what the (score, index) pair construction
already implies (experts are enumerated in index order).

**Consequence**: with a total order, every correct selection algorithm
— partial_sort, nth_element, a parallel radix select, a heap — produces
the same top-k set in the same order, provably. The implementation-
independence the descent row needed is exactly this.

## The measurement that gates the numerics question

**Tie frequency over the real serving run**: run the keep-quant paged
serving recipe and count exact f32 ties among the router scores (a
one-line counter in the instrumented router: ties within the top-k
boundary, and ties anywhere). Outcomes:

- **Zero ties across the serving run** → the comparator change is a
  PROVABLE NO-OP on every real input: the serving gate must stay
  UNMOVED (53/60, 0 hard, worst 2.486), and the tie-order adjudication
  dissolves — nothing to decide.
- **Ties occur** → the tie-break choice changes the selected expert on
  those inputs: that is the numerics adjudication (lowest-index-wins vs
  the unspecified status quo), decided by the operator with the
  measured tie count and the affected positions as the evidence. The
  serving gate may move on tie inputs; that is a re-adjudication, not a
  widening.

## The unlocked performance lever

With the total order landed, the parallel top-k is provable: any
correct algorithm agrees with the serial one. The row lands the
parallel/radix top-k (a shared-seam implementation — a host parallel
top-k; the device SigmoidBiasTopK pattern is reference) with an
equivalence gate: the parallel selection equals the serial selection
over randomized + adversarial score distributions (heavy ties included)
— which the total order makes well-posed. The FLINT-CPU-DESCENT-2
falsification harness (`test_flint_router_topk_equiv`) becomes the
equivalence gate: its inputs now have a unique correct answer.

The router top-k was 2.6% @16t doubling to ~5% @32t (the serial
partial_sort doubles at 32t) — the parallel lever is worth ~2-5% at
serving concurrency, plus it removes the serial bottleneck as
concurrency grows.

## Gates

- The tie-frequency counter: recorded per serving leg.
- The keep-quant serving gate: UNMOVED in the zero-tie case; re-pinned
  with the measured record in the ties-occur case (re-adjudication
  flagged for the operator).
- The new equivalence gate: parallel == serial over the adversarial
  corpus (unique answers by total order).
- The dtype bit-identity suite, MiMoV2 sink gates, test_flint_paged
  208/208, test_flint_gguf — the standing set, all green.

## Measured baseline

The router phase: 2.6% @16t, ~5% @32t (the partial_sort doubles at
32t). The FLINT-CPU-DESCENT-2 falsification harness
(`test_flint_router_topk_equiv`) exists — its inputs are the
adversarial corpus.

## Git integration

One pull request, `row/FLINT-ROUTER-TOTALORDER`, spec committed first:
the comparator commit, the tie-frequency evidence commit, the parallel
top-k commit (or your coherent split).

## Now

`IMPLEMENTING` — the comparator (f88dcfa12), the tie-frequency measurement
(96fbb5335), and the parallel top-k (baf44d49e) have landed on
`row/FLINT-ROUTER-TOTALORDER`; the serving gate is UNMOVED (53/60, 0 hard,
worst 2.48648); ties occur in the real serving run (2 boundary ties per
~900k router calls), so the lowest-index pin is a measured re-adjudication
FLAGGED for the operator in
`docs/bench-evidence/flint-router-totalorder-20261101/README.md`.
