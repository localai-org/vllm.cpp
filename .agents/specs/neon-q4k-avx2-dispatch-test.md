# NEON Q4_K dispatch test after AVX2 integration

## Scope

Adapt the incoming NEON dispatch test to the composed x86 AVX2 and Arm NEON selectors. Product kernels and numerical tolerances do not change.

Issue: ISSUE-LOCAL-01M6ARHFZQEEEPCH0M9JP4Y7F9

## Upstream anchors

Upstream 2359732b3563fc60bf216ee39c09b098c8e1eadb adds `tests/vt/test_neon_q4k_dot.cpp`. Its NEON-only dispatch expectation predates the PR's x86 selector.

## Design

Compare production dispatch against both selectors in production order. Keep the entire NEON bit-identity corpus. Verify the NEON selector against its explicit getter only when that getter exists. Compare traits output bitwise with the selected dispatch for all hosts. On scalar and NEON paths, retain the independent portable bit comparison. The existing AVX2 quant-dot suite supplies its independent arithmetic oracle and tolerance.

## Risks

A selector test can compare an implementation with itself. Explicit pointer checks establish reachability, while the unchanged NEON corpus and existing AVX2 suite check arithmetic independently.

## Tests and gates

Capture the default x86 failure before editing. Run the NEON suite under default, portable and forced AVX2 selection. Run the AVX2 quant-dot suite under those selections. A fresh reviewer mutates production dispatch and verifies failure. Run repository preflight and report existing failures separately.

## Evidence

The pre-fix default run at merge 17bd9ef0bc109a32166f25c2ba2905b74dd138a1 fails three assertions in the dispatch case. `/tmp/merge-avx2-neon.log` records the failure.

## Stop conditions

Stop for product arithmetic changes, unavailable architecture coverage, or an unresolved selector contract.

## Now

Implementation complete. Fresh review is pending. Arm execution requires a DotProd host and remains unavailable on this x86 host.

## Outcome

On 1 November 2026, the adapted NEON suite passed default and forced AVX2 selection with 5 cases and 260 assertions each. Portable selection passed 5 cases and 516 assertions. The independent quant-dot suite passed 35 cases and 285691 assertions per selection. Product defaults and arithmetic do not change. Arm execution remains unavailable on this x86 host. The repository preflight retains the pre-existing stale `KERNEL-ATTN-DFLASH-BLOCK` anchor failure. Normal archive linking exceeds disk quota, so focused tests link the same compiled objects through a thin archive.
