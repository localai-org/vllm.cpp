# FIX-TT-NUMA-DEGRADE: the refusal fallback is load-bearing (red/green evidence)

Date: 2026-10-29. Host: aarch64, 128 cores, 255 GB RAM, 4 NUMA nodes — this
host GRANTS `set_mempolicy(MPOL_INTERLEAVE)`, so the local red had to be
manufactured by mutation (the red itself is the CI record). Tree:
`row/FIX-TT-NUMA-DEGRADE`, worktree `/tmp/tt-numa-degrade`; mutation scratch
copy `/tmp/tt-numa-degrade-mutation` with its own build
`/tmp/build-tt-numa-mutation`; unmutated build `/tmp/build-tt-numa`. Both
configured `-DCMAKE_BUILD_TYPE=Release`, CPU-only
(`VLLM_CPP_CUDA/METAL/VULKAN/HIP/TENSTORRENT=OFF`), tests on, HF download off,
ccache. Checkpoint `/mnt/models/Aleph-Alpha/Kolibri-1` (mounted). Issue:
ISSUE-LOCAL-01M62YS0ZP0F5PC7YJFJ8CP91A. Spec:
`.agents/specs/fix-tt-numa-degrade.md`.

## RED — the CI record

The ubuntu runner image rollout left the CI sandbox refusing
`set_mempolicy(MPOL_INTERLEAVE)`. The old guard threw at `kolibri1_numa.h:142`
(`VT_CHECK`) and reded four suites that construct `PolicyGuard` —
`test_kolibri1` and `test_kolibri1_bf16_arm` and `test_kolibri1_w2` through
`LoadKolibri1Weights` (`kolibri1_weights.cpp:287`), `test_kolibri1_numa`
directly — CI run `40712283192`, job `122900454192` (first seen
2026-10-28T05:37Z, run `40639035531`). This host grants the interleave, so
there is no local red; the mutation below is the local proof that the new
degradation path carries the behavior the runners need.

## GREEN — unmutated suites on this host

The landed code, knob unset (best-effort, and the interleave is granted so it
engages): all four suites pass.

| suite | result |
|---|---|
| test_kolibri1_numa | 6 cases, 6 passed |
| test_kolibri1_bf16_arm | 10 cases, 10 passed |
| test_kolibri1_w2 | 7 cases, 7 passed |
| test_kolibri1 | 27 cases, 27 passed |

## The mutation

`detail::SetMempolicy` in the SCRATCH copy only, patched to fail
unconditionally with the runner refusal class:

```diff
 inline int SetMempolicy(int mode, const unsigned long* nodemask,
                         unsigned long maxnode) {
-  return static_cast<int>(
-      syscall(SYS_set_mempolicy, mode, nodemask, maxnode));
+  (void)mode; (void)nodemask; (void)maxnode;  // MUTATION (FIX-TT-NUMA-DEGRADE
+  // local proof): fail unconditionally, the runner refusal class.
+  errno = EPERM;
+  return -1;
 }
```

The worktree file was never touched by the mutation session (its mtime is the
last implementation edit, and `diff worktree scratch` is exactly the patch
above — so scratch minus patch reproduces the worktree byte-for-byte).

## 01-patched-four-suites.log — green under the mutation, knob unset

The four suites on the patched build with `VT_KOLIBRI1_NUMA_INTERLEAVE`
UNSET (the CI runner's shape): all four PASS. Every guard construction prints
the named warning exactly once, to stderr:

```
kolibri1_numa: VT_KOLIBRI1_NUMA_INTERLEAVE: set_mempolicy(MPOL_INTERLEAVE) failed (errno 1): interleave refused by the host; continuing with the default policy
```

| suite | cases | stderr warnings | named skips |
|---|---|---|---|
| test_kolibri1_numa | 6 passed | 4 (4 guard constructions) | 4 |
| test_kolibri1_bf16_arm | 10 passed | 63 | 0 |
| test_kolibri1_w2 | 7 passed | 1420 | 0 |
| test_kolibri1 | 27 passed | 4 | 0 |

- A count of zero would mean the warning is unreachable; more than one per
  construction would be noisy. Each construction emits exactly one warning by
  construction (the warning is the fallback arm's single `fprintf`).
- In `test_kolibri1_numa` the four interleave-only cases (guard engages,
  pages span nodes, dequant-cache arena, loader pages) each print the named
  SKIP and pass; the two both-modes cases (knob `0` disables, outside-guard
  first-touch stays on one node) run unconditionally in both modes — 5
  assertions still execute under the mutation.
- Reading the merged log: stderr is unbuffered, stdout block-buffered under a
  pipe, so warning lines cluster before the doctest banner of their suite.
  `grep -c 'VT_KOLIBRI1_NUMA_INTERLEAVE: set_mempolicy'` per suite section
  gives the stderr counts above (the SKIP messages embed the same phrase and
  are stdout, which is why a naive phrase count doubles the numa row).

## 02-patched-strict-arm.log — the `1` arm still throws by name

The same patched build with `VT_KOLIBRI1_NUMA_INTERLEAVE=1`: the guard throws
the OLD message, by name, at both a unit site and the production loader site
(exit 1 in both runs — the strict arm is by design a refusal→throw; these are
supplementary runs, not the gate runs, which are the unset ones above):

```
ERROR: test case THREW exception: vt: kolibri1_numa: set_mempolicy(MPOL_INTERLEAVE) failed at ./src/vllm/model_executor/models/kolibri1_numa.h:173
```

Recorded as filtered runs (`-tc`) of the engage case and the loader case.
Note: an unfiltered strict run throws only in the first guard case — the
knob-off case's `unsetenv` leaves the variable unset for the rest of the
process, so later cases run best-effort (the same in-process env leak the
original subcase layout already had; the knob-off case balances its own
set/unset and no assertion depends on cross-case env state).

## Test-shape note for the reviewer

The first implementation used SUBCASEs; doctest 2.5.2 suppresses every
sibling SUBCASE once one of them returns early, so a subcase-level skip
silently dropped the knob-off and first-touch cases under the mutation (the
suite still reported green). The landed test splits the four assertions into
TEST_CASEs, where a TEST_CASE-level return skips only itself — the
convention the loader case already used. Verified: under the mutation, all
six cases run (4 named skips, both-modes cases execute their assertions).

## What did NOT change

No allocation-path change: the fallback sets `active_ = false`, the dtor
returns without touching the policy, and every allocation in scope runs the
default-policy path exactly — the pre-NUMA-row path. The numerics gates are
untouched and pass unmutated (the tables above; the W3 anchor coverage in the
spec's stop condition).

> EVIDENCE-LOST (2026-10-30): `01-patched-four-suites.log`, `02-patched-strict-arm.log`
> The raw logs named above are unrecoverable: the pre-2026-10-30 global `*.log` rule
> dropped them at `git add` and the worktree removal took the untracked files; the reviews
> verified them at review time. Recorded 2026-10-30 by FIX-EVIDENCE-LOG-GATE.
