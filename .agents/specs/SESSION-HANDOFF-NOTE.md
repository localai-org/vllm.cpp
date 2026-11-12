Session handoff note — append to the implementer brief on resume:

The FLINT-TT-DENSE-MLP-STAGING row is dispatched but NOT started: the box
carried a foreign compile at dispatch time (portage-user cc1plus, load 46.9,
three 99% cpus on the bench zone) and the row's diagnosis half is
idle-gated. The implementer session picks up here:

1. Build first (does not need the bench zone): /tmp/build-p150staging,
   taskset -c 64-127, -j 64, -DVLLM_CPP_TENSTORRENT=ON + the tt-metal
   CMAKE_PREFIX_PATH from the W1/W2 evidence records.
2. Instrument: env-gated per-tensor rel_rms dumps of the dense gate/up/down
   staging at L0 and L46 vs the CPU shadow, behind a DumpOn()-style gate
   following the W1 methodology (docs/bench-evidence/flint-tt-w1-*/).
3. Attribute: per-tensor rel_rms under (a) landed order, (b) reversed
   order, (c) extended DBuf lifetimes — three legs, three orders, one
   question: does the outlier follow the ORDER (staging defect, fix in
   flint_tt.cpp) or the TENSOR (upload-path defect, repro + record, STOP
   per the spec)?
4. Fix per the attribution: (order) explicit lifetime/commit boundaries on
   the dense staging, kolibri1 host-free pattern precedent; (tensor)
   repro + record + STOP (substrate/upload defect, not in-tree).
5. Verify: W0 band gate holds under EVERY lifetime permutation the fix
   permits; single-layer probe 64/64; test_flint_paged 208/208;
   test_flint_gguf serving gate; MiMoV2 sink gates; dtype bit-identity.
6. Measured outcome: first-token rel_rms spread collapses under the fix.
7. Evidence: docs/bench-evidence/flint-tt-dense-mlp-staging-20261101/
   (README + raw logs; the claims-file line is already DONE-closed from
   the prior session — re-open it RUNNING for the duration).

Rules: mutex exec 9>$HOME/gpu.lock; flock -x 9 before device legs; env
LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium
TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium
TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin; build taskset -c
64-127 -j 64; bench legs taskset -c 32-63; governor logged per leg; NO
timeout guards; trailers block LAST exactly:
FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:zai/glm-5-3-flash [maki]
Push to the `mine` remote EXPLICITLY (never origin). Idle gate: 1-min
load < 16 + ps aux --sort=-%cpu before every measured leg.
