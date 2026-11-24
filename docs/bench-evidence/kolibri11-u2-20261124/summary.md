# Kolibri-1.1 U2 evidence (2026-11-24)

Row: `MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm`.
Issue: ISSUE-LOCAL-01M861BE22E3VJNM13ZFW19NX1. No token-parity claim.

## Gates

- Focused `test_kolibri11_u2`: **7 cases / 89 assertions / SUCCESS**
  (`focused-green.log`), after adding the named sparse-forward refusal case.
- Related ctest `test_kolibri1`, `test_kolibri11_gguf_guard`,
  `test_kolibri11_u2`: **3/3 passed** (`related-ctest.log`). The 1.0 W1
  KV-group and parse tests still pass, so the new path is byte-neutral for 1.0.

## Mutation proofs (each restored byte-for-byte)

| Mutation | Result |
|---|---|
| Remove `indexed_attention` from the allowed layer kinds | 5 cases / 3 assertions RED (`mutation-remove-indexed-kind.log`) |
| Force `IsDenseLayer(l)` false in tensor enumeration | 1 case / 1 assertion RED (`mutation-remove-dense-prefix.log`) |
| Replace `KvSourceLayer(i)` with `i` in KV group construction | 1 case / 2 assertions RED (`mutation-remove-kv-source.log`) |
| Disable the sparse-forward refusal policy | 1 case / 1 assertion RED (`mutation-remove-forward-refusal.log`) |

The third mutation initially stayed green. Investigation showed the first
implementation accumulated every layer into two broad full/SWA buckets, so
"sharing" was only co-membership. It was corrected to build **one group per KV
source**, and the test now asserts `kv_cache_groups.size() > 2` as well as
source/consumer co-membership. The mutation is red after the correction.

## Pre-existing failure, not a regression

`test_kolibri1_gguf` (the 1.0 Q2_K token gate) fails on this host with
`worst topk diff 3.96888 > band 2.5`. The same test built and run from the
unmodified base `92ea7d3a8` fails **byte-identically** (compare
`preexisting-base-kolibri1-gguf.log` with `u2-branch-kolibri1-gguf.log`; the
only differences are ctest wrapper lines). It is therefore a pre-existing
environment/gate-band condition, not introduced by U2.

## Scope note

These are parse, tensor-accounting, KV-group and named-refusal gates only.
There is no gateable CPU oracle for `kolibri1_1`; U3's sparse kernel and NVFP4
KV cache remain owed, and the forward refuses them by name.
