# Flint on Tenstorrent — W2: the ttnn::matmul f32-to-tf32 operand truncation audit

Row: BACKEND-TENSTORRENT.
Issue: ISSUE-LOCAL-01M5T8070WQ8RJYXZXD5ZWN5KZ.
Prior waves: W0 (flint-tenstorrent.md — the device arm and the 2.58e-1
baseline), W1 (tt-rmsnorm-repair — the rms_norm attribution FALSIFIED:
per-op exact norms moved the Flint rel_rms 2.579e-1 -> 2.568e-1, kolibri
unchanged).

## Scope

MEASURE FIRST, then fix only what the measurement indicts.

1. AUDIT (no code changes): enumerate every matmul in
   `src/vllm/model_executor/models/flint_tt.cpp` with its operand dtypes
   (activation f32 or bf16, weight bf16, output f32 or bf16) and identify
   every f32-OPERAND matmul that truncates to tf32 on this pin
   (tenstorrent_ops.cpp b67b30fd record). Compute the expected per-op
   bound (~2^-11 relative on a truncated operand) and the compounding
   site count across the 49 layers.
2. MEASURE the truncation in isolation: a device micro-probe comparing a
   [rows,K]x[K,N] matmul with f32 operands vs bf16 operands (identical
   bf16-rounded inputs) against a host-double reference. RED = the
   f32-operand arm's measured deviation vs host, quantified.
3. FIX CANDIDATES, picked by measurement:
   a. Round activations to bf16 before matmuls whose CPU reference also
      rounds there (the CPU forward's rounding points are the contract).
   b. Where the CPU reference keeps f32 (router / lm_head score paths),
      route through an f32-exact composition if one exists on this pin,
      else record the truncation as the open substrate defect with its
      measured bound.
   c. If the audit shows matmul truncation cannot explain the 2.58e-1
      (e.g. activations are already bf16 at every matmul), SAY SO with
      the numbers and bisect the forward site-by-site at layer 0 on a
      short prompt (embed -> norms -> q/k/v/o -> mlp -> router ->
      combine) until the first diverging op. THE BISECTION IS THE
      DELIVERABLE if the hypothesis dies; no forced fix.

## Upstream anchors

- ttnn::operations::matmul::matmul, pinned substrate
  `/tmp/umdtrial-install` / TT_METAL_HOME=~/Sources/tt/tt-metal-pin.
- tenstorrent_ops.cpp:60-93 (F32AccKernelConfig, MatmulKernel), the gdn
  comment at :1294 (sum = tf32 tile-reduce, mean = f32 SFPU path).

## Design

Static audit first: in the committed flint_tt.cpp every `vt::Matmul`
activation operand is a bf16 DBuf (`normed_dev`, `ao_dev`, `normed2_dev`,
`tok_in`, the silu-mul outputs `a`/`sa`, `fnorm_dev`); only OUTPUT dtypes
differ (bf16 everywhere except the router and lm_head, which are f32
out). tf32 truncation is an f32-OPERAND effect; a bf16 operand (8-bit
mantissa) is representable in tf32 (10-bit mantissa) exactly, and the
f32-out arms pass `fp32_dest_acc_en`. If the audit holds, the micro-probe
still runs (it quantifies the bound the substrate record names and proves
the bf16 arm clean vs host), and the wave becomes the bisection.

## Risks

- The micro-probe scratch pattern has a staging-bug history (the W0 probe
  NaN/1e30 incident: a K-element source under a T*K-byte copy). The probe
  re-derives shapes from the tensors it stages.
- The device is a shared local P150; every run under the flock mutex.
- W1's repair is NOT in this base (base 3090f069b predates
  mine/row/TT-RMSNORM-W1); the audit numbers are taken on THIS base and
  the report says so.

## Tests

- Micro-probe: scratch test in the kolibri1 b2ii pattern (or a W2 scratch
  test), comparing device arms vs host double on identical inputs.
- Bisection harness: a scratch dump of per-stage device-vs-CPU distance
  at layer 0 (embed sum, post-attn sum, post-mlp sum, per-op norms) on
  the W0 prompt; the W0 test's existing FLINT_TT_W0_DUMP=1 /
  FLINT_CPU_DUMP=1 legs plus added scratch probes, NOT committed unless a
  gate change is warranted.
- GATE: `tests/test_flint_tt_w0` with
  `FLINT_TT_W0_MODEL=/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`,
  before/after recorded honestly. If a fix moves the number, ALSO run the
  kolibri device token gate
  (`VT_KOLIBRI1_TT_B2II_MODEL=/mnt/models/Aleph-Alpha/Kolibri-1`,
  `-tc="*device token gate*"`) for the no-regress leg, and tightening the
  W0 band to the new number is IN scope.

## Gates and evidence

Evidence in `docs/bench-evidence/flint-tt-w2-20261025/report.md`: the
audit table, the micro numbers, before/after, the exact build/run recipe,
environment, and contention state. Device discipline: local P150 non-fleet,
`exec 9>$HOME/gpu.lock; flock -x 9` around EVERY device run, reset before
the first run, pinned substrate env (TT_METAL_RUNTIME_ROOT,
TT_METAL_HOME, LD_LIBRARY_PATH per the W0 report).

## Stop conditions

- NEEDS_CONTEXT if the model artifact or the device is unavailable.
- No push, no PR from this wave; the branch reports back for review.
- If the bisection indicts a site outside this wave's scope (substrate or
  CPU reference), the deliverable is the located divergence + a new issue,
  not a cross-scope fix.
