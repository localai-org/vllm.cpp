ID: ISSUE-LOCAL-01M6M9VMVN9JP3XSG7BW8RZTKN
Title: The router selection is unspecified on exact f32 ties -- one tie-break makes it a total order, unlocking the provable parallel top-k
Row: FLINT-ROUTER-TOTALORDER
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-04
Updated: 2026-11-04
Closed: -

## Problem

The Flint router selects top-6-of-384 via std::partial_sort with a comparator on score only -- among exact f32 ties the selected expert set is implementation-defined, and the router feeds the gather. FLINT-CPU-DESCENT-2 stopped honestly: a parallel replacement cannot be proven bit-identical to an unspecified order. But the (score, index) pairs already carry the index: a tie-break (lowest index wins) makes the comparator a TOTAL order, after which every correct selection algorithm produces the same result provably. The tie frequency over the real serving run gates the numerics question: zero ties = the comparator change is a provable no-op (serving gate unmoved, adjudication dissolves); ties occur = the tie-break choice is the operator's adjudication with the measured evidence. The unlocked lever: the parallel/radix top-k (2.6% @16t doubling to ~5% @32t), provable by the total order, with test_flint_router_topk_equiv as the equivalence gate. Spec: .agents/specs/flint-router-totalorder.md

## Resolution

-
