# TT-EMBED-DUAL-RESIDENCY — W3: the OOM cleared, the captured arm diverges (STOP)

Evidence for row `TT-EMBED-DUAL-RESIDENCY` (issue
`ISSUE-LOCAL-01M7CFF6ESGHZ2WQDA74PG4VDY`). Implementation heads: W1
`881752c9b`, W2 `be416cc5a`, W3 fix uncommitted at capture time and then
committed with this evidence. Build: `/tmp/build-embed` (same recipe as
`tt-staging-footprint-20261113/README.md`). Recipe per leg: mutex
`${GPU_LOCK:-$HOME/gpu.lock}`, luwen reset + 15 s + `/dev/shm` cleanup,
idle gate load<16, `taskset -c 32-63` runs / `64-127` builds, governor/
env/ps-top logged (`*-env.txt`, `ps-top-*.txt`), bench run FROM THE REPO
ROOT (kernel-JIT resolves `./src/vt/tenstorrent/kernels` from cwd — the
first three legs, `q6kl-cap-*.log` of 13:44-13:49, failed on exactly
that and were rerun).

## W3 gate result: FAILED — captured-vs-eager divergence, STOP

UD-Q6_K_XL (`701d8fa9…`), KV 8, captured arm, defaults, 3 reps:

- `q6kl-cap-{1,2,3}`: exit 0, NO OOM (the ~80 MB trace region now
  allocates), token ids 3/3 identical, sha256 `d54b29f9…` —
  **deterministic but WRONG**: stream `220,16,16,220,16,…` (the embed
  contribution collapsed to the attractor), vs the recorded eager stream
  `b36bc61a…` = `220,16,23066,220,16,…`.
- Eager arm with the same W3 tree: `q6kl-eager-1`, sha256 `b36bc61a…` —
  BYTE-IDENTICAL to the recorded TT-STAGING-FOOTPRINT eager stream, Mean
  TPOT 5340.30 ms (recorded 5340.63).

## The W3 fix is coherent and not the cause — the regression is W2's

The uncommitted W3 work converts the committed tensor to the hidden
slot's CURRENT dtype+layout before every in-place copy (eager
`EmbeddingKernel` and captured `EmbedDeviceIdsInto`), because the
slot's shadow dtype churns between ops (f32 after the embed commit, bf16
after a layer op) and `ttnn::copy`'s dtype-conversion path only supports
TILE inputs; the row shadow itself is now staged F32 (bf16-rounded
values upcast — the twin's exact bits, zero conversions anywhere).

Bisection: Q4_K_M at `--num-blocks 64` (the TT-STAGING-FOOTPRINT gate-2
recipe, recorded captured stream `d49c53d2…`):

| head | captured sha256 | result |
|---|---|---|
| W3 tree | `a0033dd8…` | diverges (`q4km-cap-1-token-ids.json`) |
| W2 `be416cc5a` (scratch build `/tmp/build-bisect`, `bisect-q4km.log`) | `a0033dd8…` | diverges identically |
| pre-change `de3493a3742` | `d49c53d2…` | recorded pass |

So the captured quant-row branch introduced in W2 `be416cc5a` breaks the
captured arm on BOTH artifacts; the W3 dtype fix neither caused nor
repairs it. The eager arm is correct at every head. The captured path
reads the F32 row shadow in-region and still produces wrong rows —
next hypotheses: the in-region `typecast`/`to_layout` ops recorded at
capture time replay against buffers the per-step H2D refresh does not
update (the refresh writes the ROW_MAJOR f32 shadow; the recorded
converted copy is a different buffer), or the refresh ordering vs replay
launch. `q6kl-cap-pure-1`, `q6kl-cap-sync-1` token files and the
13:44 failed-cwd logs are kept as bisection residue.

## What PASSED (the twin elimination is real)

- DRAM ledger (`VT_TT_ALLOC_TRACE=1`, `q6kl-cap-1.stderr.log`):
  before (TT-STAGING-FOOTPRINT oom8) — 13,638,720 B/bank free at
  `end_trace_capture`, 80,134,144 B trace region OOM. After — run
  completes, tightest snapshot `total_free=2,633,993,216 B`
  (~329 MB/bank). Freed ≈ 2.5 GB/bank-equivalent — the 2.44 GiB twin.
- Focused: `test_gguf_keep_quant` 12205/12205 (`/tmp/kq.log`), green
  with the W3 tree; eager-arm byte-identity `b36bc61a…` reproduced.
- TPOT captured 5337.25 / 5317.94 / 5318.44 ms vs eager 5340.30 ms —
  capture costs nothing measurable (measured on the WRONG stream, so
  recorded as indicative only, not a gate result).

## Owed (row stays OPEN)

- Root-cause the W2 captured quant-row divergence (STOP hit; W3(b)-(d)
  — KV16 re-measure, Q8_K_XL viability — not run past the STOP).
- Fresh implementer + reviewer cycle on the repair.
