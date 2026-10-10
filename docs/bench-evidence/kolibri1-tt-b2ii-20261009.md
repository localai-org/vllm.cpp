# Kolibri-1 Tenstorrent — B2b-ii: the streaming MoE (2026-10-09)

Row MODEL-TEXT-kolibri-1-tenstorrent, slice B2b-ii (spec
`.agents/specs/kolibri-tt.md` ### B2 scope — B2b addendum, slice ii).
Issue ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN. Branch
`row/kolibri-tt-b2ii` (base 6b63e62c6 = origin/main).

## 1. What landed

- `kolibri1_tt_stream.h/.cpp` — the HOST half: the slot-pool plan (per
  layer capacity = the B1 `hot_experts` capped by the per-layer residual
  share AND the expert domain; the policy built UNCHANGED through
  `PlanKolibri1TTExpertSlotPolicy`), the fetch executor's host half
  (remap, fetch jobs, byte accounting), the per-step stream-bound guard
  (LOUD refusal), the slot shadow (the readback pivot's reference side;
  no `extract_shard` on this tt-metal), the eviction-hook integration,
  and the `Kolibri1TTSlotEpoch` reset-lane recording.
- `kolibri1_tt_forward.h/.cpp` — the DEVICE arm: the slot pool staged on
  the card (FP8_E4M3 packed bytes verbatim, row-major, concatenated
  gate/up/down; f32 scale grids stay host-side), the routed path in
  `MoeBlock` (dispatch -> fetch -> stage -> readback-verify -> `Touch()`
  -> memoized slot dequants -> the CPU row's gather/ExpertMlp/scatter ->
  `vt::MoeCombine`), the reset lane clearing the memo on
  `ContentChangedSince`, and the B1 per-token stream bound charged per
  layer with a LOUD refusal past the ceiling. The B2b-i refusal remains
  ONLY for the streaming-disabled arm (`VT_KOLIBRI1_TT_B2II_STREAM=0`)
  and genuine over-capacity/miss-handle failures, by name.
- `kolibri1_registry.cpp` — the TT dispatch arm attaches the streaming
  context; the env check is read per call so both arms exercise in one
  process.
- Tests: `test_kolibri1_tt_b2ii.cpp` (9 host cases + the device token
  gate leg, W3 methodology) and the b2bi device-leg contract moved to
  slice ii (routed path carries the decode, refusal silent, streaming
  counters asserted).
- Spec: `## Now` / `## Owed` updated; the pin reference corrected to
  `vllm-cpp-pin/20261008 @ a585e5744a8`.

## 2. Build recipe

```
cmake /tmp/vllm-kolibri-tt-b2ii -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_TENSTORRENT=ON \
  -DCMAKE_PREFIX_PATH="$HOME/Sources/tt/tt-metal-pin/build_Release/lib64/cmake;$HOME/Sources/tt/tt-metal-pin/build_Release/share/cmake"
ninja test_kolibri1_tt_b2ii test_kolibri1_tt_b2bi
run: TT_METAL_RUNTIME_ROOT=$HOME/Sources/tt/tt-metal-pin \
     LD_LIBRARY_PATH=$HOME/Sources/tt/tt-metal-pin/build_Release/lib64:.../libexec/tt-metalium
```

## 3. Host-side gate results (red-first, no card)

`test_kolibri1_tt_b2ii`: **9/9 cases, SUCCESS** (canonical durable-lib
build, 2026-10-09). The RED was the new TU's own build/step failure
before `kolibri1_tt_stream.cpp` existed; the refusal cases were each
observed failing for their stated reason (message content asserted) and
the positive cases pin the exact byte/slot contracts:

- slot-pool plan: per-layer capacity, packed 3x2048 + scales 12 B on the
  tiny fixture, policy fields consumed unchanged, total pool <= residual.
- pool refusals: "ONE slot per layer" and "host tier does not equal" both
  throw by name.
- fetch list: remap (resident vs distinct misses), job slot/offset
  (`slot * packed_bytes_per_slot`), host payload identity (the loaded
  weights' first packed projection base), `stream_bytes == misses x
  expert_bytes`, repeat-collapsed requests fetch once.
- LOUD stream bound: dispatch over the ceiling and a guard charge past it
  both throw naming "per-token bound"/"stream bound exceeded" + "LOUDLY".
- slot shadow: byte-exact verify; corrupted readback refuses
  ("DEVICE READBACK ... byte-for-byte"); the eviction hook clears the
  evicted slot's shadow (FilledSlots decrements, `Has` false).
- reset lane: a slot swap flips `ContentChangedSince` (resets == 1), an
  identical re-selection does not (resets == 0).
- router readback pivot: whole-buffer download contract; non-positive
  count refuses.

B2a planner contracts stay green unchanged: `test_kolibri1_tt`,
`test_kolibri1_tt_b2i`, `test_kolibri1_tt_b2bi` host half — 4/4 ctest
green (canonical build, 2026-10-09). CPU battery in the TT build:
`test_kolibri1`, `test_kolibri1_w2`, `test_kolibri1_moe_glue`,
`test_kolibri1_dequant` green. `test_kolibri1_dequant_cache`: the
documented fork() EAGAIN flake reproduces on the CLEAN BASE in this
window (pid >= 0 at line 335, 6/7) — retried isolated three times over
~40 min, still failing; pre-existing, not this branch (verified by
`git stash` baseline run).

## 4. Device window — BLOCKED, external (stop condition engaged)

The device legs (smoke, token gate, bench anchor) DID NOT RUN. The P150
device path is broken at the board/driver level on this host right now:

- Symptom: `TT_FATAL mesh_device.cpp:874: cq_id 0 is out of range` on
  device ops — including ops that were GREEN yesterday (the b2i
  embedding/staging gate, the vt cast/embedding health trio).
- Reproduces on the UNCHANGED base commit 6b63e62c6 (baseline worktree
  built and run — same crash), so it is NOT this branch.
- Reproduces on BOTH the durable in-tree lib64 (a585e5744a8, rebuilt
  2026-10-08 21:45) AND a fully-consistent /tmp/pin-build 6449cf13f7b
  libs+runtime combination.
- The b2i evidence doc's documented recovery loop (kill stuck procs;
  `luwen reset` under `$HOME/gpu.lock` x3, `tt-smi -r 0` + 15 s, metal
  cache clear) was executed and does NOT recover.
- dmesg shows the card was PCI-rescanned (`tenstorrent 0002:01:00.0:
  Found a Tenstorrent Blackhole device ... enabling device (0000 ->
  0002)`); `/dev/tenstorrent/` now carries node `0` beside `by-id/`.
  The board likely needs a driver reload or host reboot — an operator
  action, not a slice fix.
- Per the addendum's stop conditions the device work is recorded and
  OWED; the row stays `ACTIVE`. No card measurement was attempted past
  the blocker, and none is recorded here.

## 5. Owed (device)

- Device smoke: routed path live, slot fills + readback verification on
  card, stream bound asserted, swaps exercising the reset predicate.
- THE TOKEN GATE: golden replay on the TT device path, 141/145 argmax
  with flips inside the 2.5-nat band, 0 hard flips, per-flip nat gaps
  (the gate leg is committed in `test_kolibri1_tt_b2ii.cpp`, env
  `VT_KOLIBRI1_TT_B2II_MODEL`).
- Bench anchor (only after the token gate) per the recorded recipe.
- dram_free before/after slot-pool staging, once the card window opens.

## 6. Device window, second session (2026-10-09, pin-forward stack)

Branch `row/ci-tt-pin-forward` (base origin/main + the tt-metal
pin-forward port, PR #3430). Stack: tt-metal pin main + tt_umd 0.9.12,
KMD 2.11.1-pre, FW 19.15.0, tt-smi 6.7.0. Model
`/mnt/models/Aleph-Alpha/Kolibri-1`. Clean kernel cache before every
measured leg; `tt-smi -r 0` + 15 s + cache/shm clear before device legs.

What landed this session (commits on `row/ci-tt-pin-forward`):

- the native `kMoeCombine` TT kernel (host-staged, bit-exact to the CPU
  oracle; red-first test 961/961 green);
- the token gate's own defect fix: it decoded step 0 from a zeroed KV
  cache while claiming the prompt positions computed (the gate leg had
  never reached numerics before, so this was its first exposure); the W3
  walk is now one token per step — a whole-prompt prefill step also
  exceeds the B1 per-step stream bound (1242865920 B charged against
  1179936000 B at layer 15);
- the f32-shadow RmsNorm arm: the pinned tt-metal's device bf16
  `rms_norm` returned EXACTLY ZERO for the model's per-head q/k norms —
  attention output 0.0 at every layer, 0/8 hard flips with nonsense
  tokens. After the fix the attention stream matches the CPU row
  (layer-0 attn_n sum 17.25 TT vs 17.23 CPU, the CPU row over identical
  inputs).

Gate result (W3 methodology, `test_kolibri1_tt_b2ii` device leg):

| config | compared | match | flips | hard | per-flip nat gaps |
|---|---|---|---|---|---|
| host-free decode ON (default) | 8 | 0 | 8 | 8 | 6.6-12.6 (nonsense tokens) |
| `VT_TT_HOST_FREE_DECODE=0` | 33 | 26 | 7 | 7 | 2.69, 2.91, 3.20, 4.02, 4.32, 5.60 (near-ties, semantic tokens) |

VERDICT: the gate is FAILING and stays open — 141/145 with 0 hard flips
is the contract, and no configuration reaches it. With host-materialized
KV/rope handoffs the model is numerically CLOSE to the CPU row (every
flip is a genuine semantic near-tie just past the band); with the
host-free device paths the activations corrupt. The gap to the band is
the same order as the two open op-level drifts.

Op-level verdicts (both reproduce on the clean cache):

- `kGdnDecode` wide-range state: state max_abs 0.0051074 vs tol 0.002,
  rel_rms 3.6e-5 (the error concentrates in small elements, max_rel 81).
  Mechanism: the composed step's `dot = sum(Sd*krow)` reduction — the
  eltwise multiply is exact (2e-9 vs a double oracle); the reduction
  loses ~4e-5 relative on |dot| ~ 1e3, and `v' = (v - dot) * beta`
  amplifies it. A pairwise reshape-sum experiment made it WORSE
  (0.0195) and was reverted. OPEN.
- `kMatmulBTQuantGrouped` decode P=1 Q4_K: 154/1024 outputs past the
  elementwise envelope, worst_rel 0.130697, worst_abs 285 (diffs ~0.2%
  of |17e3|-scale outputs). Not exercised by the kolibri1 path (fp8
  experts, bf16 GEMMs). OPEN.
- NEW intermittent (unowned): `batched decode RAC is capture-safe
  (num_slots=2)` failed mid-suite once (V shadow 109/128 token-exact)
  and passed in isolation and on a full-suite re-run.

Owed after this session: root-cause the host-free device handoff
(device rope / device ReshapeAndCache / device residual norm are the
paths host materialization bypasses); the two op-level drifts above;
the decode bench anchor 109726 (chain 101807/109726), gated behind the
token gate per the addendum.

## 7. Substrate rms reduction-precision patch (2026-10-10, row/tt-kolibri-residue)

The rms_norm bias root-caused in section 6's red micro was fixed at the
TT-METAL SUBSTRATE level, not in vllm.cpp. The trial tt-metal tree
`/tmp/tt-metal-umdtrial` now carries a reduction-precision patch on top
of the port fixes, and `/tmp/umdtrial-install` was rebuilt from it
(build dir `/tmp/build-umdtrial2`; rebuild recipe:
`cmake --build /tmp/build-umdtrial2 -j 4 && cmake --install . --prefix
/tmp/umdtrial-install`). ANY future measurement or gate on this stack
must use this rebuilt install.

Substrate patch (3 edits in the trial tree):

- `ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp`
  (`rmsnorm_default_compute_config`): `fp32_acc = true`,
  `approx_mode = false`. Mechanism: the old default (`fp32_acc=false`)
  made the layernorm program factory run Float16_b CBs with
  `float32_reduction=false` (`layernorm_op_multi_core.cpp:499`) even
  for FLOAT32 input tiles — the kolibri1 norms' sum-of-squares
  accumulated in bf16, the whole -3.4% bias.
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/
  layernorm_op_multi_core.cpp`: the reduce scaler CB is Float32 when
  `fp32_dest_acc_en` (the 1/W mean factor no longer rides a bf16 tile).

Verdicts (same env/reset/cache recipe as section 6, clean reset per leg):

- Micro (`SCRATCH dbg rmsnorm micro`, bit-identical bf16 [1,2560]
  seed 7): sum ratio 0.98992 -> 0.99972 vs the CPU row, max_abs
  0.0625 -> 0.015625 (4 ULP -> 1 ULP). Against the host-double sum
  rounded per element to bf16, TT 0.99915 vs CPU 0.99942: the residual
  is the bf16 output-store quantization floor, not a kernel bias.
- B2b-ii gate, host-free ON: ARGMAX CHAIN 26/33, flips 7 (7 hard);
  instrument 40 tf flips (5 near-tie, 35 HARD), worst 3.19 nats
  (baseline 47 HARD, worst 5.17).
- B2b-ii gate, `VT_TT_HOST_FREE_DECODE=0`: ARGMAX CHAIN 26/33, flips 7
  (1 near-tie, 6 hard — baseline 7 hard); instrument 116 tf flips
  (105 HARD), worst 7.59 nats, deterministic across a clean reset.
- Verdict: the systematic norm bias is gone but the argmax chain did
  not move: the remaining flips belong to the two open op-level drifts
  (kGdnDecode, kMatmulBTQuantGrouped), both re-confirmed failing in
  isolation on this stack. The >=141 target stays open.
- `test_tenstorrent_backend` on this stack: the only cases failing in
  isolation are the two known ones; the W4 EnsureDevice2D
  `uploads_bulk_bf16` counter and the matmul region class split fail
  only in full-suite order state and pass isolated (same intermittent
  class as the section-6 RAC flake).
