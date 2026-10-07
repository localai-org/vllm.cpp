ID: ISSUE-LOCAL-01M3JM3HMK2F41A10P7T2WQXD1
Title: the a7c23ac96d parity pin advanced in 4f11dfc10 but six restating surfaces kept e126687a9a, and the gate reads only the span
Row: ENG-RECORD-CLAIM-AGREEMENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

The vLLM parity pin advanced in 4f11dfc10 (PR #3320, merged 2026-09-26) from e126687a9a828d513c01a07cd69f025f27d63280 to a7c23ac96d7806e7c7e7d862eadbce5a33529b94, and the authority block in .agents/upstream-sync.md moved with it. Nothing else did. check-oracle-pins went red with 11 errors and every one of them was the SAME defect at a different site: the oracle record copy and the five prose surfaces named in PIN_SURFACES still carried the old revision.

Updating the seven pin-bearing values turns the gate green and leaves the tree asserting things that are now false, which is worse than the red. Concretely, after the spans move:

1. .agents/NOW.md:25 reads "Pin: vLLM a7c23ac96d (0.3.0.dev267) since 2026-09-03 (#2817). A gate HAS now run at it and it PASSED (2026-09-04, job 7386f034..., dgx:gpu0)". No gate has run at a7c23ac96d. The capture was taken at e126687a9a on 2026-09-04 and #2817 is the issue for THAT advance, not this one. The date, the issue reference and the "at it" all bind to the wrong advance once the pin in the sentence moves.

2. .agents/oracles/vllm.md, the whole of "## What this pin establishes, and what it does NOT", is written about e126687a9a: the thor build block names e126687a9a828d51... explicitly, item 2 says the declared token-exact gate at "this pin" was captured 2026-09-04, and item 3 owes step 6 at that pin. None of it was re-measured at a7c23ac96d. The 2026-09-22 sync report for this advance records exactly two gates, test_scheduler 48/48 and the vllm library build, and both are OUR tree, not the oracle. So the section keeps its heading and its contents while the pin above it now names a revision for which it establishes nothing.

3. The dgx:gpu0 row of the device-scoped table still reads "**e126687a9a, the CURRENT pin**". It is not the current pin; it is the prior one. The two strix rows read "**5559679229, the PRIOR pin**", which after this advance is two pins back, and the residual paragraph under the table says "Whoever adds the next row at e126687a9a narrows it further" when the next row belongs at a7c23ac96d.

4. docs/FEATURES.md:14 says "the parity pin since 2026-09-03" and :45 says the current pin "advanced on 2026-09-03". Both dates describe the e126687 advance. The same paragraph cites NOW.md as recording "**NO gate has run at it**" - a sentence NOW.md no longer contains, and which becomes TRUE again only once the "at it" in NOW.md is re-pointed at the pin that actually has no gate.

5. docs/benchmarks/how-we-measure.md:21, speculative-decoding.md:4 and vllm-online-serving.md:15 all date the advance 2026-09-03 and cite #2817, and all three describe FlashInfer as moving to 0.6.18 "at the new pin". The authority block still records flashinfer_version = 0.6.15.post1 after the advance, and .agents/oracles/vllm.md item 3 records the 0.6.15.post1 to 0.6.18 step as NOT discharged. So the substitution would attach an unmeasured dependency to a pin nobody has built.

The class is the one this row exists for: a fact corrected in one place and left standing in another, and here the correcting place is the one the gate reads, so the gate goes green over a tree that says the pin moved and, everywhere else, that a gate ran at it. Fixing only the spans is the change that makes the defect invisible.

## Resolution

-
