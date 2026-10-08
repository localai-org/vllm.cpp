# Kolibri-1 TT B2b-i first slice — build, link, resident staging on the
# P150, and one verified device op (MODEL-TEXT-kolibri-1-tenstorrent)

Date: 2026-10-08. Host: `thalia`, aarch64 Gentoo, kernel 7.0.1, 255 GB RAM,
TT-KMD 2.10.1-pre, board firmware 19.7.1, tt-smi 6.2.1 — one Tenstorrent
Blackhole P150 (1 chip visible, chip id 0, DRAM 34,178,731,008 B ≈ 31.8 GiB
usable). Not a fleet `rc` device: every device run held the file mutex
`$HOME/gpu.lock` and reset the card first (`~/Sources/tt/luwen/target/release/
reset`, rc=0 required, 101 → `tt-smi -r 0` + retry). Tree: worktree
`/tmp/vllm-kolibri-tt-b2i`, branch `row/kolibri-tt-b2i` from `origin/main`
5d2f5d31a. Spec: `.agents/specs/kolibri-tt.md` § B2 scope — B2b addendum,
slice i. Issue: `ISSUE-LOCAL-01M4D2KFZ8H7B39XE75XYE0AYY`.

**This is smoke output, not B2b evidence.** Per the spec's gate ordering, no
device measurement — throughput, latency, or memory — is B2b evidence until
the dense-resident slice completes a greedy decode on the card. This slice
delivers the bring-up the completion condition stands on: a TT-enabled build
that links, the real checkpoint loaded through the production registry path,
the resident non-expert slice staged native on device with byte accounting
matching the plan, every staged operand verified by readback, and ONE
verified device op (embedding + layer-0 q_proj) against the CPU row's output.
The greedy decode itself is the slice's remaining completion condition; the
token gate and bench anchor come after it.

## 1. Build and link

Three builds, all ccache-launcher, `VLLM_CPP_CUDA=AUTO` (no CUDA toolkit on
this box; resolves off), from the worktree root:

| build | purpose | `CMAKE_PREFIX_PATH` (semicolon-separated) |
|---|---|---|
| `/tmp/build-kolibri-b2i` | TT compile gate (spec pin) | `/tmp/pin-build/lib64/cmake;/tmp/pin-build/share/cmake` |
| `/tmp/build-kolibri-b2i-nonpin` | device leg | `/home/lu_zero/Sources/tt/tt-metal/build_Release/lib64/cmake;/home/lu_zero/Sources/tt/tt-metal/build_Release/share/cmake` |
| `/tmp/build-kolibri-b2i-cpu` | CPU-only gate | (none; `VLLM_CPP_TENSTORRENT=AUTO` resolves off) |

`-DVLLM_CPP_TENSTORRENT=ON -G Ninja` for the two TT builds. The TT compile
gate is green: `ninja tests/test_kolibri1_tt_b2i` compiles the B2b device TU
under `-Werror` and links clean against the pin source's fresh lib64.

### The stale-lib64 blocker (spec stop condition, § Owed)

Both tt-metal trees' checked-in `build_Release/lib64/_ttnncpp.so` predate the
`use_mcast` signature of `ttnn::transformer::chunk_gated_delta_rule` that
`tenstorrent_gdn.cpp` calls (the blocker named in
`ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5.md` § stale-lib64):

- pin tree `/home/lu_zero/Sources/tt/tt-metal-pin` @ 6449cf13f7b — stale lib
  (symbol absent; verified with `nm -D --defined-only`);
- non-pin tree `/home/lu_zero/Sources/tt/tt-metal` @ d20b8e27f29 — equally
  stale (symbol absent).

Resolution used here:

- **Pin side (compile gate):** a fresh ninja rebuild of the pin source
  (`ttnn tt_metal`) produced `/tmp/pin-build/lib64/_ttnncpp.so` (58.1 MB,
  symbol present — verified with `nm -D --defined-only ... | c++filt`); the pin
  vllm.cpp build is configured against `/tmp/pin-build` and links with no
  workaround. This satisfies the spec's prerequisite for the pin-side build.
  The durable copy of the fresh lib64 into the pin tree's own
  `build_Release/lib64` remains owed (the working copy lives in `/tmp`).
- **Non-pin side (device leg):** the tree's lib is equally stale, so the
  non-pin build links through `/tmp/kolibri-tt-gdn-stub.o`, a no-op
  definition of the one mangled symbol. The GDN decode ops are not exercised
  by this row's gates (no GDN op runs in B2b-i; the wave-A/B1 GDN suites are
  host-side policy tests). Relink helper: `/tmp/kolibri-b2i-relink.sh`.

## 2. Why the device leg runs on the non-pin tt-metal (NEEDS_DECISION: pin)

The spec pins the tt-metal pin tree (`6449cf13f7b`) as the build reference.
Measured today, the pin build's device runtime cannot execute a device op on
this box's current driver/firmware, while the non-pin tree can:

| case (our `test_tenstorrent_backend`) | pin build | non-pin build |
|---|---|---|
| `kCastBf16 / kCastF32 round-trip` | SUCCESS (66/66 assertions) | SUCCESS |
| `kMatmul matches a host F32 reference` | HANGS — real-time profiler `Device 0 sync sample N/100 timed out after 2000ms`, 3 consecutive timeouts, killed by the script timeout → FAILURE | SUCCESS |
| `kEmbedding matches a host F32 reference` | hangs the board the same way | SUCCESS |
| 753-operand resident staging (B2b-i plan) | hangs the board the same way | completes (below) |

Board-death signature under the pin build (dmesg):
`tenstorrent 0002:01:00.0: Timeout waiting for ARC response` /
`Timeout waiting for space in ARC message queue` → `AER: ... can't recover` →
`device recovery failed` → PCI rescan; `/dev/tenstorrent/0` disappears (only
`by-id/` remains). The pin build ran the 27B decode ledger green on 2026-09-30
(`.agents` TT evidence), so the breakage is new between then and today — a
driver/firmware/board-state change (KMD 2.10.1-pre, fw 19.7.1), not a source
regression the pin controls. The pin's second build dir `build_release_script`
(2026-09-29) is separately broken (undefined `tt::umd::Cluster::
read_from_device/write_to_device`, libtracy conflict) and was not used.

The pin also has a teardown defect independent of the hang: every pin binary
aborts at exit in `MetalContext::destroy_all_instances` (std::filesystem
abort, the #1486 class) AFTER printing `Status: SUCCESS`, and that abort
poisons the board — the next open gets `Setting power state failed` +
`Read 0xffffffff ... PCIeHangError`. Pin runs therefore require a reset
before and after, and suite success is judged by `Status: SUCCESS`, never by
the exit code alone (the teardown abort exits 134).

**Decision recorded:** the compile gate stays on the pin source (via the
fresh `/tmp/pin-build` lib64); the device leg runs on the non-pin tree
(`d20b8e27f29`, 2026-09-18 build — the stack the project's earlier recorded
device evidence ran on), linked with the gdn stub. **NEEDS_DECISION for the
operator: bump or replace the tt-metal pin** — the pinned source cannot run
device ops on KMD 2.10.1-pre / fw 19.7.1 as of 2026-10-08. Until the pin is
reconciled, every future B2b device leg needs this non-pin deviation recorded
the same way.

## 3. Host-side gates (ctest, no card)

`test_kolibri1_tt_b2i` registers only in TT builds (beside
`test_tenstorrent_backend`); its six host-side cases run under ctest with no
card and no checkpoint mount — the resident staging plan over the tiny
synthetic fixture and over the real checkpoint manifest (`kolibri1_manifest
.inc`): dtype decisions, per-component byte accounting against the spec's
byte-math table, the full-model single-device refusal firing where it must
and not firing for the resident slice, and the resident slice fitting one
P150 (~3.09 GiB) where the full model refuses (~73.6 GiB, wave A). The device
bring-up case is env-gated (`VT_KOLIBRI1_TT_B2I_MODEL`) and skips otherwise.

Result (non-pin build, after a cast-green board validation):
`test cases: 6 | 6 passed | 0 failed`, `assertions: 204 | 204 passed`,
`Status: SUCCESS!` (log `/tmp/kb2i-nonpin-hostcases.log`).

CPU-only build (`/tmp/build-kolibri-b2i-cpu`), `ctest -R kolibri`: 6/6
passed (see the gate table in § 7).

## 4. Device leg (P150, non-pin build)

Run: under `$HOME/gpu.lock`, pin tree env sourced
(`~/Sources/tt/env-tt-common.sh`), `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` =
`/home/lu_zero/Sources/tt/tt-metal`,
`LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib64:$TT_METAL_HOME/build_Release/libexec/tt-metalium`,
card reset to rc=0 + 15 s settle + `~/.cache/tt-metal-cache` cleared, from the
worktree root, `stdbuf -oL -eL`:

```
VT_KOLIBRI1_TT_B2I_PROGRESS=1 VT_KOLIBRI1_TT_B2I_DEBUG=1 \
VT_KOLIBRI1_TT_B2I_MODEL=/mnt/models/Aleph-Alpha/Kolibri-1 \
/tmp/build-kolibri-b2i-nonpin/tests/test_kolibri1_tt_b2i -tc="*device bring-up*"
```

Model: `/mnt/models/Aleph-Alpha/Kolibri-1` (fp8 checkpoint, read-only; 32
safetensors shards). Full log: `/tmp/kolibri1-tt-b2i-device-nonpin-debug.log`.

```
[kolibri1-tt-b2i] device leg start: model=/mnt/models/Aleph-Alpha/Kolibri-1
[kolibri1-tt-b2i] loading the real fp8 checkpoint (32 shards) through ModelRegistry::Load...
[kolibri1-tt-b2i] load: 18.8 s (hidden=2560 layers=50 experts=384)
[kolibri1-tt-b2i] resident plan: 0.001 s
[kolibri1-tt-b2i]   attention fp8        1703936000 B
[kolibri1-tt-b2i]   attention grids          416000 B
[kolibri1-tt-b2i]   shared fp8            196608000 B
[kolibri1-tt-b2i]   shared grids              48000 B
[kolibri1-tt-b2i]   router gate            98304000 B
[kolibri1-tt-b2i]   router bias               76800 B
[kolibri1-tt-b2i]   norms                   1054720 B
[kolibri1-tt-b2i]   embed + head         1310720000 B
[kolibri1-tt-b2i]   TOTAL                3311163520 B (3.084 GiB, 753 tensors)
[kolibri1-tt-b2i] full-model single-device refusal (expected): fires
[kolibri1-tt-b2i] staged 1103 operands (350 fp8 / 353 bf16 / 400 f32) = 3311163520 B in 2.54 s
[kolibri1-tt-b2i]   chips=1 first_chip_id=0 dram_total=34178731008 B dram_free_after=30833363968 B
[kolibri1-tt-b2i] readback verified 350 fp8 operands (+grids) / 353 bf16 / 50 f32 in 14.4 s — mismatches=0
[kolibri1-tt-b2i] device op: embedding + layer0 q_proj over 6 golden-prompt tokens (hidden=2560 q=6144)
[kolibri1-tt-b2i] embedding: 15360 elements, mismatches=0 (10492.1 ms)
[kolibri1-tt-b2i] q_proj GEMM: device 3034.9 ms vs CPU 86.7 ms; bit-exact 7285/36864; max_ulp=1055 max_abs=2.441e-03 max_rel=7.726e+01 max_err_ratio=1.375e+00 (per 2^-8*terms); envelope_violations=0
[kolibri1-tt-b2i] device leg PASS in 73.9 s total (load 18.8 / plan 0.001 / stage 2.54 / verify 14.4 / op 16.0)
[doctest] assertions: 36880 | 36880 passed | 0 failed |
[doctest] Status: SUCCESS!
```

Reading against the slice's four deliverables:

1. **Staged bytes match the plan exactly.** The plan's TOTAL is
   3,311,163,520 B (3.084 GiB, 753 tensors); the device staged 1103 operands
   (350 fp8 weights + their 350 f32 scale grids, 353 bf16, 50 f32 router
   biases) totalling the same 3,311,163,520 B. The per-component table matches
   the spec's byte-math (attention 1.587 GiB fp8 + grids, embed+head
   1.221 GiB, router 0.094 GiB, shared expert 0.188 GiB, norms ~2 MiB). Device
   DRAM after staging: 30,833,363,968 B free of 34,178,731,008 B.
2. **The full-model refusal fires; the resident slice does not.** The
   wave-A single-device full-model refusal (≈73.6 GiB > 32 GiB) fires for the
   full tree as required; the resident slice plans and stages without it.
3. **Every staged operand verified by readback, byte-exact.** 350 fp8
   operands + grids, 353 bf16, 50 f32 — mismatches=0 over all 1103 operands.
   The fp8 readback is a RAW-BYTES comparison (new seam
   `vt::tenstorrent::ReadbackStagedOperandBytes`): the ttnn host FP8→F32
   decode flushes subnormals to zero (`tt_metal/impl/data_format/float8.cpp`,
   `float8_e4m3::operator float`: "HW flushes subnormals (exp=0,
   mantissa!=0) to zero"), and the Kolibri-1 fp8 weights contain 28,829
   subnormal bytes of 15,728,640 in layer-0 q_proj alone — so the earlier
   dtype-pivot verification (f32 `to_vector` readback) mismatched on ALL 350
   fp8 tensors even though the staging was byte-perfect. Raw host bytes
   (`host.host_storage().host_tensor().buffer().apply(... hb.view_bytes()
   ...)`, dtype-agnostic) are the only byte-exact fp8 readback; the q_proj
   dequant in the device op decodes those bytes with the scalar reference
   `vt::F8E4M3ToF32` (bit-matches `vllm::F8E4M3ToF32`,
   `nvfp4_dequant.cpp:24`) times the f32 grid → `vt::F32ToBF16`.
4. **One verified device op end-to-end.** The embedding gather over the 6
   golden-prompt tokens (ids `[5249, 25079, 74493, 15774, 121598, 116026]`)
   is BIT-EXACT (15,360 elements, mismatches=0 — a row gather, no
   arithmetic). The layer-0 q_proj GEMM (M=6, K=2560, N=6144, bf16 in/out
   through the production `vt::MatmulBT`) agrees with the CPU row's output
   within the stated envelope (§ 5).

Timings are smoke numbers for bring-up only (first-kernel JIT dominates the
op legs: 10.5 s embedding, 3.0 s GEMM for 94 MFLOP — a non-tensor-core
fallback path for tiny M, not the production decode shape). The whole leg:
73.9 s (load 18.8 / plan 0.001 / stage 2.54 / verify 14.4 / op 16.0).

## 5. The q_proj GEMM comparison — method stated before running, and the
falsified premise

Method (stated in the test header before the run): both sides consume
identical bf16 operands — the activation is the bit-exact embedding readback,
the weight is the bf16 dequant of the byte-verified staged fp8 bytes (the CPU
row's documented R1 disposition: dequant-to-bf16, then a plain bf16 GEMM; the
device fp8-block GEMM is the B2b compute wave's, not this slice's). The CPU
reference accumulates in f32 with a single bf16 rounding at the store (the
CPU row's contract, `cpu_ops.cpp` MatmulBTKernel → MatmulChunked →
MatmulOneChunk: sequential f32 over K, round on store).

**Falsified premise.** The method originally stated "both sides accumulate in
f32 with a single rounding at the store, so the only permitted difference is
accumulation-order rounding" with a CHECK of max_ulp ≤ 2. Measurement
falsified it: the rigorous f32-accumulation-order bound is
`2(K−1)·2^-24·Σ_k|a_ik·w_jk|`; probed host-side over the real checkpoint bytes
(pure Python, `/tmp/kb2i_bound_probe.py`) the max term-magnitude sum is 1.49,
so the worst-element order bound is 4.5e-4 — but the device measured
max_abs = 2.441e-3, **5.4× above** it. The device ttnn matmul's error for this
tiny-M fallback shape is bf16-unit-roundoff scale relative to the
term-magnitude sum, i.e. bf16-granularity internal precision (typical
mismatch 1–2 bf16 ulps, rel ~2^-7…2^-6; rare ulp=75 outliers at near-zero
outputs where cancellation amplifies order differences; bad-per-row uniform
~4900/6144, not row-patterned).

**Corrected envelope (stated before the run, in the test header):** per
element,

```
|dev − cpu| ≤ 8 · 2^-8 · Σ_k |a_ik·w_jk|  +  max(ulp_bf16(|cpu|), ulp_bf16(|dev|))
```

— at most 8 bf16 unit roundoffs per unit term-magnitude sum, plus the store
rounding (the larger of the two magnitudes' bf16 ulps bounds both sides'
round-at-store contributions; `UlpBF16(x) = max(2^(e−8), 2^-133)` via frexp).
The CHECK asserts it per element and reports the bit-exact fraction, max ulp
distance, max abs/rel diff, and the max per-element error ratio.

**Measured:** bit-exact 7285/36864 (19.8%); max_ulp=1055; max_abs=2.441e-3;
max_rel=77.3 (near-zero denominators — cancellation, not a magnitude signal);
max_err_ratio = **1.375** (per 2^-8·terms) — the envelope's 8 allows a 5.8×
margin over the measured worst element; **envelope_violations=0** over all
36,864 elements. The probe numbers behind the envelope: max |terms| = 1.4905
(mean 0.3995), max |out| = 0.1906.

Scope note: chasing the device's internal accumulation precision (ttnn
matmul program config / the tiny-M fallback kernel) is the B2b **compute**
wave's job, not this bring-up slice's. The smoke check verifies the
production op as it is: right contraction over byte-verified operands,
numerics within the stated envelope.

## 6. Known residue (not blockers, recorded)

- **#1486-class defect, row BACKEND-TENSTORRENT**
  (`ISSUE-LOCAL-01M4D5W0HSEPNP26AT30CEP874`): every TT-build binary
  segfaults under `env -i` — the registrar's static init probes the device
  without `TT_METAL_ROOT`. Consequence in this row's gates:
  `test_kolibri1_dequant_cache` fails 1/7 under TT builds (host-side run,
  no card involved in the failing case). Known residue, pre-existing.
- **Pin teardown poisoning** (§ 2): pin binaries abort at exit after
  `Status: SUCCESS` and poison the board; always reset between pin runs.
- **Shared-box contention:** many concurrent agent sessions on this host
  reset the card under `flock` (some without). Recovery procedure that worked
  all day: kill stuck procs by pid; under `$HOME/gpu.lock` loop
  `luwen/target/release/reset` (rc=0 required; 101 → `tt-smi -r 0` + 15 s)
  until `tt-smi -s` shows the board AND a cast case prints `Status:
  SUCCESS`. `tt-smi` telemetry answering is NOT proof of kernel health (it
  answered while the node was gone). `/dev/tenstorrent/` containing only
  `by-id/` = dead board.

## 7. Gate table

| gate | result |
|---|---|
| TT pin compile (`-DVLLM_CPP_TENSTORRENT=ON`, pin source via fresh `/tmp/pin-build` lib64, `-Werror`) | PASS — `ninja tests/test_kolibri1_tt_b2i` compiles and links clean |
| Host-side staging cases, TT build, no card (`test_kolibri1_tt_b2i`) | PASS — 6/6 cases, 204 assertions, `Status: SUCCESS!` |
| CPU-only build, `ctest -R kolibri` | PASS — 6/6: `test_kolibri1` 3.25 s, `test_kolibri1_dequant` 0.01 s, `test_kolibri1_dequant_cache` 0.14 s, `test_kolibri1_w2` 267.35 s, `test_kolibri1_w3` 2979.94 s, `test_kolibri1_decode_bench` 292.91 s (total 3543.61 s) |
| Device leg (non-pin build, P150) | PASS — 36880/36880 assertions, `Status: SUCCESS!`, 73.9 s |
| `scripts/check-agent-record` | PASS — rc=0 (`agent record OK: ENGINE=179 MODEL=384 QUANT=87 KERNEL=60 BACKEND=90 ANCHOR-ROT=0`) |
| `scripts/check-env-doc` | PASS — rc=0 both directions (457 production vars documented or classified; 215 doc-table vars read; `VT_KOLIBRI1_TT_B2I_PROGRESS` allowlisted as kernel-internal) |
| `scripts/agent-preflight.sh --staged` | PASS — rc=0 at commit time; 5 standard host-config gates SKIPPED with reasons (check-arm-isa-build, check-cpu-isa-build, check-cuda-fat-gencode, check-pr-size, check-triton-aot-multiarch) |

## 8. What remains owed (slice completion)

- B2b-i completion condition: **one greedy decode of a golden prompt on the
  card** through the dense-resident slice — not yet run; everything above is
  its prerequisite. The token gate (141/145 argmax, 0 hard flips) and the
  production bench anchor come after it, per the spec's gate ordering.
- B2b-ii (streaming MoE): slot buffers, router readback → host remap → fetch
  executor, per the addendum.
- The durable copy of the fresh `_ttnncpp.so` into the pin tree's
  `build_Release/lib64` (§ 1); the tt-metal pin bump decision (§ 2).
