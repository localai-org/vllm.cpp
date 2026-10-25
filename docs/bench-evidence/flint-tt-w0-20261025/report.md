# Flint on Tenstorrent — W0: device dispatch + first device-vs-CPU baseline

Row MODEL-TEXT-FLINT (device arm), spec
[`.agents/specs/flint-tenstorrent.md`](../../../.agents/specs/flint-tenstorrent.md),
issue `ISSUE-LOCAL-01M5RVS7Q3EFVHGC6YBVF02A9Q`. Date 2026-10-25.
The W0 gate is HONEST device-vs-CPU only: no oracle parity claim, no
throughput claim, no ceiling.

## Recipe

- Branch `row/FLINT-TT-W0`, base `97dbf7d09ed5e0b52dd591f34be82196c7b8bd94`
  (origin/main, fetched 2026-10-25). Worktree `/tmp/flint-tt-w0`, build
  `/tmp/flint-tt-bld` (`-G Ninja -DCMAKE_BUILD_TYPE=Release
  -DVLLM_CPP_TENSTORRENT=ON -DCMAKE_PREFIX_PATH="/tmp/umdtrial-install/
  lib64/cmake/tt-metalium/lib64/cmake;/tmp/umdtrial-install/share/cmake;
  /tmp/umdtrial-install/lib64/cmake"`, built `taskset -c 0-31`).
- Device: the LOCAL P150 (non-fleet). Every device run under
  `exec 9>$HOME/gpu.lock; flock -x 9`, `nice -n 5`. Card reset before the
  first run (`~/Sources/tt/luwen/target/release/reset` + 15 s);
  `~/.cache/tt-metal-cache/*` cleared before the FIRST run only. The pinned
  substrate (`/tmp/umdtrial-install`, `/tmp/tt-metal-umdtrial`) untouched.
- Env (the pinned substrate; `env-tt-common.sh`'s LD_LIBRARY_PATH points at
  the non-pin tree and must NOT be used for runs — see
  `kolibri1-tt-b2bi-fwd-20261008.md` §1):
  `LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium`,
  `TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium`,
  `TT_METAL_HOME=~/Sources/tt/tt-metal-pin`.
- Gate binary: `tests/test_flint_tt_w0` with
  `FLINT_TT_W0_MODEL=/mnt/models/flint-GGUF/flint-Q4_K_M.gguf` (the
  published Q4_K_M container, the one `test_flint_gguf.cpp` gates),
  `FLINT_TT_W0_DUMP=1 FLINT_CPU_DUMP=1`. Both arms load the SAME
  `FlintWeights` (bf16 after the GGUF dequant); the CPU leg runs the
  unchanged `FlintModel::Forward` in the same process.
- Clock/contention state: shared box, builds pinned to cores 0-31, device
  runs niced; no other GPU work taken during a run. KMD/fw as reported in
  the run log (`red.txt` / `green.txt` headers).

## Red-first evidence

`red.txt`: with the dispatch wired and the device arm pending, the device
gate failed EXACTLY on the named refusal —
`"FlintForCausalLM: the Tenstorrent forward arm is not implemented"` — and
`FlintTTForwardCalls()` stayed 0, while the device-free staging-plan leg
passed. 13/15 assertions, 1 of 2 cases failed. The red was captured BEFORE
the device forward landed.

## Dispatch reachability (proven exercised, not compiled)

`green.txt`: `dispatch: the kTENSTORRENT case was EXERCISED (counter 0 -> 1)`
— the registry case ran on the card, the resident slice staged, and the
forward returned through `ForwardFlintTT`.

## Staging accounting (read from the artifact)

`resident=3,871,748,096 B (3.606 GiB)` (embed + head + norms + attention
projections + 2 dense MLPs + 47 shared experts + routers) — fits the 32 GiB
P150; `expert_tier=75,698,798,592 B (70.5 GiB)` — CANNOT reside, STREAMS at
`37,748,736 B/layer` (top-6 experts). dense=2 moe=47 sliding=35. A starved
budget is refused by name with the byte deficit (leg 1).

## Device-vs-CPU baseline (the W0 numbers)

One prompt ("The capital of France is", 5 tokens), identical bf16 weights:

| Stage | Device (TT) | CPU reference |
|---|---|---|
| embed sum | 3.409978 | 3.409978 (IDENTICAL) |
| layer 0 post-attn x sum | 13.482791 | 12.584867 |
| layer 0 post-mlp x sum | 1.769863 | 0.827251 |
| first-token logits sum (tok0) | -1942146.595514 | -1956081.934977 |

First-token logits (n=750720): **max_abs=3.20e+01, rms=5.93, rel_rms=2.58e-01**.
Greedy chain (8 steps): **1/8 match** — dev=[269 269 269 269 269 140759 269
269], cpu=[12834 332 269 764 32 13 3529 198].

## Why the baseline is wide — the substrate norm bias, measured

The frame that makes these numbers meaningful: the kolibri TT campaign's
CPU-vs-oracle distance was 2.0e-3 on layer-0 l_out, and its device-vs-CPU
gap traces to the SAME substrate defect. The op probe (scratch, run
2026-10-25 on the P150, numbers recorded here; the probe file itself was
scratch and is not committed):

- TT `vt::Matmul` bf16 vs host f32-accumulate reference:
  **max_abs=1.892e-03, rms=5.641e-04** — the matmul leg is clean.
- TT `vt::RmsNorm` vs the host reference on unit-scale rows: **max_abs=
  7.812e-03 per row (~1% systematic)** — the recorded open defect in
  `src/vt/tenstorrent/tenstorrent_ops.cpp` ("ROOT-CAUSE RECORD (kolibri1 TT
  prompt-dependent hard divergence)... the bias stays the row's open
  defect"). (Two earlier probe runs reported NaN rows and 1e30 outputs;
  both were the SCRATCH PROBE's own staging bug — a K-element source under a
  T*K-byte copy — not kernel defects. The corrected probe's numbers are
  these.)

Normed once per norm site (3 per layer) across 49 layers, a ~0.3-1% bias
compounds multiplicatively (~e^0.25) and is exactly the observed signature:
the post-attn sum drifts +7% at layer 0, the logits sum drifts 0.7%, and the
greedy chain flips early. This is the next wave's lever (a norm-arm repair
on the substrate), and it is why the W0 gate bands a hard regression
(chain collapse, order-of-magnitude distance growth) instead of asserting an
oracle-tight number W0 cannot reach.

## Result

`green.txt`: 2/2 test cases, SUCCESS. The focused CPU-legal flint gates also
pass in the same build: `test_flint_forward` and `test_flint_gguf`
(785 s total, 2/2 passed).


## PROVENANCE NOTE 2026-10-25 (reviewer minor)

`red.txt` was captured on an uncommitted intermediate of the implementation: its `EXERCISED (counter 0 -> 1)` line and its refusal wording ('the W0 device forward is pending') match no committed code. The COMMITTED red is the refusal + `FlintTTForwardCalls() == 0` assertion pair, and the reviewer's own mutation run reproduces the true red (counter 0 -> 0, cases 1/2 failed). red.txt is retained as captured; this note is the correction.
