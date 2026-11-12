Row FLINT-TT-DENSE-MLP-STAGING — session handoff (2026-11-01)

Status: DIAGNOSE STAGE NOT STARTED — the session's context budget was
exhausted during orientation. The worktree is at the row's FIRST commit
only: the spec (.agents/specs/flint-tt-dense-mlp-staging.md) and the
issue (ISSUE-LOCAL-01M6BQY53TJ3PPEVT7DT472HQP.md), pushed to
mine/row/FLINT-TT-DENSE-MLP-STAGING. No instrumentation, no evidence, no
fix has been written yet.

What the next session owes, in order (this is the whole task):

1. Instrument flint_tt.cpp: env-gated per-tensor rel_rms dumps of the
   dense gate/up/down staging at L0 AND L46, under (a) landed order,
   (b) reversed staging order, (c) extended DBuf lifetimes — three legs,
   same weights, same box, taskset -c 32-63, governor logged.
2. Attribution verdict: does the outlier follow the ORDER (staging
   defect) or the TENSOR (upload-path defect)? This is the row's core
   question — answer it with numbers before touching anything.
3. FIX per the attribution: (order) explicit lifetime/commit boundaries
   on the dense gate/up/down staging — the kolibri1 host-free pattern is
   the house precedent; (tensor) STOP — that is a substrate/upload bug,
   repro + record, not an in-tree repair.
4. VERIFY: the W0 model-level band gate must hold under EVERY DBuf
   lifetime permutation the fix permits (the spread must collapse).
   Plus: single-layer attention probe still 64/64 per-head slices (both
   window modes); test_flint_paged 208/208; test_flint_gguf serving gate
   unmoved; MiMoV2 sink gates; dtype-dispatch bit-identity.
5. Evidence: docs/bench-evidence/flint-tt-dense-mlp-staging-20261101/
   (README + raw logs, plain git add — the whitelist is live).

Device recipe (mandatory, every device leg): mutex exec 9>$HOME/gpu.lock;
flock -x 9; reset ~/Sources/tt/luwen/target/release/reset + sleep 15 +
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory; env
LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium
TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium
TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin; build taskset -c
64-127 -j 64, /tmp/build-p150staging; serving legs taskset -c 32-63;
governor logged per leg. NO timeout guards anywhere.

Gates: check-agent-record; check-env-doc (new envs); commit style
--range origin/main..HEAD; trailers --range --filled. Push to `mine`
EXPLICITLY — never origin (mudler's repo; leaks deleted twice).

Stop conditions: a substrate-level defect (reproduces outside
flint_tt.cpp) → repro + record + STOP; a fix requiring a W0 band
re-adjudication → land only with measured before/after rel_rms +
reviewer confirmation it is a re-pin, not a widening.

House discipline: builds taskset -c 64-127 -j 64; bench legs taskset -c
32-63; idle gate load < 16 + ps aux --sort=-%cpu before every measured
leg; claims-file RUNNING/DONE; governor logged per leg; trailers block
LAST; never origin.
