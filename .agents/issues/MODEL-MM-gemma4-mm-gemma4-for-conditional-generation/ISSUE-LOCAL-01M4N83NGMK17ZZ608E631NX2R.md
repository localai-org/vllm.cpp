ID: ISSUE-LOCAL-01M4N83NGMK17ZZ608E631NX2R
Title: Discriminate vision pooling rounding and excess weight storage
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-11
Closed: 2026-10-11

## Problem

Review of 3c799f892 finds two test coverage gaps, not known production bugs: pool-no-round survives width4 scale2 and attention >= size survives absent oversized cases. Add independent non-power-of-two pooling and exact shape refusals; prove both compiled mutations fail.

## Resolution

2026-10-10: Test coverage repair adds independent width12 sqrt12 exact pooling and 34 tower size plus nine raw connector refusal assertions. Real compiled pool-no-round and oversized-weights mutants exit1 (one and four assertions); baseline vision5/1172. Four targets and metadata28 pass. Evidence assets/work/2026-10-10-e4b-vision-repair/README.md. Production unchanged; root rerun and fresh review pending. Full preflight prohibited/PENDING.
