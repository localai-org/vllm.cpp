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
