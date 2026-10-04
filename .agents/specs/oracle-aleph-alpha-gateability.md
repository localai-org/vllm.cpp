# ORACLE-ALEPH-ALPHA-GATEABILITY: the aleph-alpha-inference pin builds and runs Kolibri-1, or it says why not

Row: `ORACLE-ALEPH-ALPHA-GATEABILITY`.
Kind: record gateability measurement.

**Secondary/primary oracle:** `aleph-alpha-inference` (the model-author's own
out-of-tree vLLM plugin; upstream vLLM implements nothing for `kolibri1`).

## Scope

`.agents/oracles/aleph-alpha-inference.md` records `gateable = no` with pin
`049a6a7bd240`. Per AGENTS.md §"Measure gateability", the oracle is gateable
only once it demonstrably builds and runs the model. This row runs that
measurement and records one of two results:

- the plugin at the pin serves `Aleph-Alpha/Kolibri-1` (78.9 GB bf16, mirrored
  at `/mnt/models/Aleph-Alpha/Kolibri-1`) and greedily decodes a fixed prompt
  set, so `gateable` becomes `yes` and `evidence` becomes a path in this tree; or
- it does not, and the record keeps `gateable = no` with the exact reason and
  the condition that would settle it.

## Owed

- [ISSUE-LOCAL-01M448RD2AEMHE0QZAEC61M0TJ](../issues/_owed/ISSUE-LOCAL-01M448RD2AEMHE0QZAEC61M0TJ.md): the gateability measurement itself.
