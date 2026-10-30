# Flint TT keep-quant dispatch gate — the device forward consumes block weights

Date: 2026-10-27. Row: MODEL-TEXT-FLINT (flint-keepquant increment). Issue:
ISSUE-LOCAL-01M5WCPZC4TWR9GZVJHH73E1XE. Spec:
`.agents/specs/flint-keepquant.md` (W-b/W-c, the ## Owed device-forward gate).

## What landed

The dispatch wiring, four pieces:

1. **Registry routing** (`src/vllm/model_executor/models/flint.cpp`): the
   GGUF case reads `VT_FLINT_KEEP_QUANT` explicitly and routes to
   `LoadFlintKeepQuantFromGguf` when `=1` (the loader self-gates too; the
   routing is now greppable at the production call site).
2. **Block staging** (`flint_tt.cpp` `StageWeight`): a block-carrying weight
   uploads its VERBATIM block bytes through `dense_attn::ResidentWeight` (raw
   byte length, cached on the OwnedTensor's `d_dev`) and is viewed `[N,K]` nk
   under the block dtype — the MatmulBTQuant orientation. Reinterpreting
   blocks as bf16 (the old `StageBf16` path) is unreachable. The routed-expert
   tier (all 256 experts x 3 projections per MoE layer) stages once at build;
   the bf16 arm's per-token host streaming path is untouched.
3. **Forward dispatch** (`MatmulWeight`): `vt::IsBlockQuant(w.dtype)` routes
   to `vt::MatmulBT` (whose hand-off reaches `kMatmulBTQuant`), else
   `vt::Matmul` — every weight site (q/k/v/o, dense MLP, router, shared
   experts, routed experts, lm_head). `FlintTTQuantDispatchCalls()` counts
   block dispatches.
4. **The CPU reference refusal**: `FlintModel::Forward` refuses a
   block-carrying tree BY NAME (the CPU arm reads bf16 elementwise); the W0
   harness's CPU reference loads the bf16 arm over the same container — the
   CPU dequantizes, the device keeps blocks.

## The gate (per-config-deterministic, one leg per config, env labeled in log)

Container: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf` (sha untested here; the
loader census below pins the shape). Device: P150, local non-fleet,
`flock -x $HOME/gpu.lock` around every run, `reset` + 15 s before the first.
NO timeout guards. Recipe per leg in the committed logs.

### Reachability mutation (THE acceptance)

Throwing on the FIRST block-carrying tensor in `MatmulWeight`'s dispatch path
(scratch copy, restored byte-for-byte after):

- MUTATED: keep-quant leg RED — `CHECK_NOTHROW` THREW "REACHABILITY
  MUTATION: block-carrying weight reached the dispatch path",
  `FlintTTQuantDispatchCalls() == 0` assertion also RED
  (`w0-mutation-red.log`, exit 1).
- RESTORED: keep-quant leg GREEN (`w0-keepquant-green.log`, and the
  post-restore rerun `w0-keepquant-green-restored.log`, exit 0).

The prior "gate" stayed green under exactly this mutation — that was the
reviewer's MAJOR. It cannot now: the forward itself throws.

### Keep-quant leg (`VT_FLINT_KEEP_QUANT=1`)

- Census: resident 1.557 GiB + expert tier 21.311 GiB RAW BLOCK BYTES,
  streamed/layer = 0; 30501 Q4_K + 5939 Q6_K tensors kept, 48 bf16 (norms,
  sinks, embed, router).
- `quant_dispatch_calls = 5138` (blocks CONSUMED by the device forward).
- First-token logits vs the CPU reference (bf16 arm, same container):
  max_abs = 3.335e+1, rms = 5.053e+0, rel_rms = 2.198e-1 (n = 750720).
- Greedy chain 0/8, dev=[260 2227 297 269 6963 29554 269 6963]
  cpu=[12834 332 269 764 32 13 3529 198] — recorded, not pinned (the
  adjudicated-divergent class; the committed chain policy).
- doctest: 2/2 cases, 19/19 assertions, exit 0.

### Distance band, and why

The keep-quant class band is the committed drift-tolerant fence, max_abs < 64
and rel_rms < 0.5. Justification: (a) the comparison is FUNCTION-level
near-tie by design — the CPU reference dequantizes to bf16 and computes in
bf16, the device keeps blocks and quantizes activations once per matmul to
the int8-dot pairing (intentionally coarser; the Qwen3.8 arm ships the same
polarity); (b) the pinned substrate's ~1% systematic rms_norm bias compounds
over 49 layers (the recorded open defect, `tenstorrent_ops.cpp` ROOT-CAUSE
RECORD), which dominates the greedy-chain flips in BOTH arms; (c) cross-epoch
drift of ~20% on the absolute rel_rms is recorded (the same committed
binaries measured 2.212e-1 and 2.683e-1 in different epochs) — this session's
keep-quant leg measured 2.198e-1 and the bf16 leg 2.683e-1 (bit-equal to the
committed evening-epoch values), both well inside 0.5. An absolute band
tighter than the drift measures the EPOCH, not the code.

### bf16 regression leg (env unset)

- Census: 3.606 GiB resident + 70.5 GiB streamed tier, 37.7 MB/layer —
  byte-identical to the committed bf16-arm numbers.
- `quant_dispatch_calls = 0` (no block dispatch on this config).
- max_abs = 3.279e+1, rel_rms = 2.683e-1, chain 2/8 — identical to the
  recorded pre-change bf16 leg (the arm is untouched).
- doctest: 2/2 cases, 18/18 assertions, exit 0.

### Kolibri no-regress leg

NOT run: no shared keepquant code changed (the change touches only
flint.cpp, flint_tt.cpp/.h, and the W0 harness). Per the spec (W-e) the leg
is owed only on a shared-keepquant change.

## Deviations

- The W0 harness CPU leg: in keep-quant mode the CPU reference now loads the
  bf16 arm (`LoadFlintFromGguf`) and runs `FlintModel::Forward` directly,
  because the registry CPU forward refuses block-carrying trees by name (the
  refusal is part of this change). The bf16 leg's CPU path is unchanged.
- First build-tree run failed with the TT kernel compiler include dir
  resolving `__FILE__` relative to the CWD (`./src/vt/tenstorrent/kernels`
  not found from `/tmp/flint-keepquant-bld`); the tests run from the source
  root, as the prior device legs did. Recorded here because the failure
  message names the CWD, not the fix.
- The mutation scratch edit was briefly lost by a `git checkout --` that
  reverted the whole file instead of only the mutation; the implementation
  was re-applied identically and the restored tree re-gated green
  (`w0-keepquant-green-restored.log`). No other file was touched by the
  checkout.

## Environment

Host: local P150 (non-fleet). env: `~/Sources/tt/env-tt-common.sh`,
`TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium`,
`TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin`,
`LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium`;
luwen `reset` + 15 s before the first run; `nice -n 10 taskset -c 0-31` on
every run; build `/tmp/flint-keepquant-bld` (ninja), tests run from
`/tmp/flint-keepquant`.

> EVIDENCE-LOST (2026-10-30): `w0-mutation-red.log`, `w0-keepquant-green.log`, `w0-keepquant-green-restored.log`
> The raw logs named above are unrecoverable: the pre-2026-10-30 global `*.log` rule
> dropped them at `git add` and the worktree removal took the untracked files; the reviews
> verified them at review time. Recorded 2026-10-30 by FIX-EVIDENCE-LOG-GATE.
