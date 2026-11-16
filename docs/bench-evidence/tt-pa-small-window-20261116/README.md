# TT-PA-SMALL-WINDOW — the paged-attention device-SDPA small-window defect: characterization and root cause

Date: 2026-11-16. Worktree `/tmp/tt-pa`, branch `row/TT-PA-SMALL-WINDOW`.
Isolated reproduction: commit `67d041d76` on `row/QWEN38-DFLASH2-STAGE-C`
(the W22 discriminator, `tests/vt/test_tenstorrent_backend.cpp`).

## The defect as reported (the stage-C W22 discriminator)

One raw RAC write + one raw PA decode: garbage at seq=2 (4.86e33) and
seq=5/7 (2.99717e+32, identical bytes), clean at seq=1/3/4/6/8/64; the
Copy-seeded arm clean at seq=1; the RAC's writes land in all 40 mix reps.

## W1 characterization (this row)

All runs on the P150, discipline per leg (mutex, luwen reset + 15 s + shm
cleanup, idle gate load < 16, taskset 32-63 runs, kernel-JIT from the repo
root, durable-libs-first, `TT_METAL_RUNTIME_ROOT=$HOME/Sources/tt/tt-metal-pin`,
INT8DOT=1). Geometry: the W22 discriminator's — Hq=4, Hkv=2, D=128,
block 32 (the engine's KV layout), 8 blocks, f32 host contract.

| Arm | Log | Result |
|---|---|---|
| RAC-seeded sweep seq 1..64, shared cache, fresh meta per seq | `w1-rac-sweep.log` | garbage at seq=1 only (1.43305e+38, bits `fed79ee3`, non-bf16); seq 2..64 clean |
| Copy-seeded sweep seq 1..64 (same layout, no RAC) | `w1-copy-sweep.log` | garbage at 31/64 seqs, bits `67600968`/`67600aa8`/`67600c48` recurring — the defect is NOT RAC-specific |
| Payload tracking (two different RAC payloads, seq 1..8 each) | `w1-payload-tracking.log` | payload A: 0 garbage; payload B: garbage at 1/2/8 — state-dependent, not payload-determined |
| Fresh cache per seq (RAC slot 0 per seq) | `w1-fresh-per-seq.log` | garbage at 2/5 — the pattern shifts with process state |
| Program cache disabled, RAC-seeded sweep | `w1-rac-sweep.log` (second case) | garbage at 1/3/6/7/8 with arg-like bytes (`3f800000`, `00000041` = the idle-core RTA marker) — worse without the cache |

Fingerprints across all arms:

1. The garbage element is ALWAYS in heads 2/3 — the second KV group (g=1).
2. The garbage bits are NEVER bf16-rounded (low 16 bits nonzero) — the
   bytes were never computed by any bf16 op.
3. The bytes are recycled memory content: RISC-V instructions
   (`00c72023 00c587b3 00f72623`), runtime-arg words (`00000041` = 65, the
   idle-core marker; `3f800000` = 1.0f), NaN patterns (`ffffffff`), and
   plausible-but-wrong previous outputs.
4. Which seq values garbage is NOT intrinsic to cur_pos — every arm shows a
   different set. Deterministic within a fixed process state; shifts with the
   allocator/program-cache history.

The stage-C discriminator's seq-parity pattern (2,5,7) is therefore a
state fingerprint of that process, not a property of the defect. The
invariants (heads 2/3, never-computed bytes, recycled content) travel.

## W2 root cause

Bisect with `VT_TT_PA_DUMP=1` (env-gated dump of the raw `sdpa_decode`
output before our commit path, `w3-rac-dump2.log` / `w3-rac-dump-fixed.log`):
**`dev_out` is clean at every call** — all elements bf16-rounded, no
suspects — including the calls whose committed output is garbage.

Conviction: the defect is OURS, in the commit path between the device SDPA
output and the host readback — NOT in the upstream tt-metal kernel, NOT in
the RAC write, NOT in the page walk.

Mechanism (`tenstorrent_paged.cpp` `TryPagedAttentionDeviceDecode`, the
identity-order commit path): `CommitDeviceLogical2D(out, flat, ...)` moved
the **bf16** `sdpa_decode` output onto the caller's output slot as the
device shadow. `EnsureHostBytes` (`tenstorrent_residency.cpp:1587-1597`)
then writes the host bytes at the **shadow's dtype width** — the FLINT
slot-serve rule: "The HOST write width is the SHADOW's dtype — the format
the content is actually in — never the slot's registered capacity." The
harness's output tensor is **f32** (`Tensor::Contiguous(mem_out, kF32, ...)`):
512 values × 2 B = 1024 B written into a 2048 B f32 buffer.

The f32 consumer then reads:

- elements 0..255 (heads 0/1): bf16 pairs reinterpreted as f32 — plausible
  values at wrong indices, small oracle deltas → reads "clean";
- elements 256..511 (heads 2/3): bytes 1024..2047 — NEVER WRITTEN —
  recycled host memory: the garbage.

The mechanism check (`w3-rac-mechcheck2.log`): re-reading the same buffer
as bf16 pairs matches the oracle to ~1e-3 (the bf16 envelope) on every
garbage row, while the f32 read shows 1e38/6.7e7 — the values are correct;
only the width was wrong.

The same violation exists in the prefill arm's device-commit path
(`CommitDeviceLogical2D` of the chunked-SDPA pieces). The host-scatter
fallbacks honor out's dtype (`StoreElemF32`) — which is why every
small-D PA test (D=8, declined by the TILE gate) passed before.

The engine's default paths pass bf16 outs (`dense_attn_block.h` `adt`,
`qwen3_dflash.cpp:1012`), so the width mismatch is latent there; the
registered `VT_QWEN3_ATTN_F32=1` f32 A/B path and any f32-out caller hit it.
The stage-C consumption gates measure whether the DFlash2 degeneracy clears
with the contract honored.

## The falsified hypotheses (recorded so they stay falsified)

- RAC write visibility/coherence: falsified — the Copy-seeded arm garbages
  too (`w1-copy-sweep.log`).
- The upstream tt-metal SDPA kernel at small windows: falsified — `dev_out`
  is clean at every call including the garbage ones (`w3-rac-dump2.log`).
- The program-cache-hit run-args class (the layernorm `d1be4cf5b`
  precedent): falsified — the defect is WORSE with the program cache
  disabled, and the buffer-bindings re-patch on every hit at this pin.
- The writer kernel's unfenced Blackhole L1-cache read of the compute-packed
  output (a patchset candidate actually applied and tested): the fence
  changed nothing — the garbage bits stayed identical (`w3-rac-sweep-fixed.log`
  vs `w1-rac-sweep.log`) — falsified and reverted in the pin tree.

## W3 the fix

`PaOutTypedForCommit` (`tenstorrent_paged.cpp`): reconcile the commit's
shadow dtype with OUT's contract — typecast the SDPA output to OUT's dtype
(f32/bf16; f16 has no ttnn dtype at this pin and declines to the host
scatter) before `CommitDeviceLogical2D`, in BOTH the decode and the prefill
arms. Capture-safe by the W4 doctrine (the eager warmup step runs the same
out dtype, warming the typecast program).

## Gates

- The focused reproduction: red-before (`w1-rac-sweep.log`, garbage at
  seq=1/64 swept) → green-after (`w3-rac-sweep-GREEN.log`, 0 garbage / 64,
  max_abs 0.002238 = the bf16 envelope).
- All five characterization arms green after the fix: `w3-all-arms-GREEN.log`
  (5 passed, 0 failed).
- The existing PA suites unchanged: `w3-pa-regression.log` (5 passed — the
  GQA reference, the device-vs-host oracle arms, the pure-prefill arm),
  `w3-rac-regression.log` (the RAC bit-exact arm).
- The [gdn]/PA failing-SETS: `docs/bench-evidence/qwen38-dflash2-stage-c-w3-20261115/gdn-failing-set.txt`
  compared on the stage-C consumption run.
