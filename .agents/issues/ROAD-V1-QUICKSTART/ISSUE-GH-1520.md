ID: ISSUE-GH-1520
Title: **`check-doc-checkpoint`'s `LANDING_SOURCE_FILES` has no entry for the page the README hands its quickstart to, so shrinking `README.md` into a pointer is refused.** The gate refuses a README claim change that touches no landing source, and the set names `.agents/mission.md`, `CMakeLists.txt`, three `benchmarks/demo/*.json` files and the two example mains. Its message names two failures, "routine checkpoints belong in the purpose-specific docs" and "co-edited public projections never justify README churn", and both are a README that GROWS. `row/DOCS-QUICKSTART-1281` is the opposite: it adds `docs/QUICKSTART.md` and cuts the README `## Quickstart` block from three command fences to a four-line pointer, so the README loses material to the purpose-specific document. At head `1b6e458c8` `scripts/agent-preflight.sh` reports 88 gates ok, zero skips and one failure, `doc-checkpoint range`, on a branch that honestly touches no member of the set. FIXED HERE by adding `docs/QUICKSTART.md` to `LANDING_SOURCE_FILES`, on its own branch with its own spec, red-first case and three mutations, because the previous implementer on the quickstart row correctly refused to widen the set inside the change it unblocks. The widening admits ONE exact path and no class: `docs/BUILD.md`, `docs/STATUS.md` and every other page still cannot license a README claim change, and `landing_page` still permits a README edit rather than demanding one. Spec [doc-checkpoint-landing-source-quickstart.md](../../specs/doc-checkpoint-landing-source-quickstart.md).
Row: ROAD-V1-QUICKSTART
State: UNKNOWN
Kind: bug
GitHub: 1520
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:535`

### Frozen archive evidence

> | [#1520](https://github.com/mudler/vllm.cpp/issues/1520) | `ROAD-V1-QUICKSTART` | **`check-doc-checkpoint`'s `LANDING_SOURCE_FILES` has no entry for the page the README hands its quickstart to, so shrinking `README.md` into a pointer is refused.** The gate refuses a README claim change that touches no landing source, and the set names `.agents/mission.md`, `CMakeLists.txt`, three `benchmarks/demo/*.json` files and the two example mains. Its message names two failures, "routine checkpoints belong in the purpose-specific docs" and "co-edited public projections never justify README churn", and both are a README that GROWS. `row/DOCS-QUICKSTART-1281` is the opposite: it adds `docs/QUICKSTART.md` and cuts the README `## Quickstart` block from three command fences to a four-line pointer, so the README loses material to the purpose-specific document. At head `1b6e458c8` `scripts/agent-preflight.sh` reports 88 gates ok, zero skips and one failure, `doc-checkpoint range`, on a branch that honestly touches no member of the set. FIXED HERE by adding `docs/QUICKSTART.md` to `LANDING_SOURCE_FILES`, on its own branch with its own spec, red-first case and three mutations, because the previous implementer on the quickstart row correctly refused to widen the set inside the change it unblocks. The widening admits ONE exact path and no class: `docs/BUILD.md`, `docs/STATUS.md` and every other page still cannot license a README claim change, and `landing_page` still permits a README edit rather than demanding one. Spec [doc-checkpoint-landing-source-quickstart.md](../../specs/doc-checkpoint-landing-source-quickstart.md). | bug |

## Resolution

-
