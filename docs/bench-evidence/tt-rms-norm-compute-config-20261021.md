# tt-metal rms_norm: the 0.9809 and 1.0072 readings differ by the substrate compute config, not by the caller (2026-10-21)

Row MODEL-TEXT-kolibri-1-tenstorrent. Issues
ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF (the pinned-side deviation) and
ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF (the upstream probe). Branch
`row/TT-RMS-K-DISCREPANCY` (base 3ce14f35c = origin/main). Device: one P150
(Blackhole) under the file mutex `${GPU_LOCK:-$HOME/gpu.lock}`. No product code
changed; this file is the record.

## 1. The question

Two replays of the same captured in-model norm rows read two different
deviations against a host-double oracle:

| row | pinned replay (2026-10-18 record) | upstream replay (2026-10-20 record) |
|---|---|---|
| `normrow-000` (1x2560, max&#124;x&#124; 3.54688) | tt/host2 = 1.0072 | tt/host2 = 0.9809 |
| `normrow-004` (1x2560, max&#124;x&#124; 5.5) | tt/host2 = 1.0077 | tt/host2 = 1.0033 |
| `normrow-001` (48x128) | tt/host2 = 0.9958 | tt/host2 = 0.9970 |

The pinned 0.9958 for `normrow-001` is this document's own measurement. The
record quotes 0.9984 for that row, and that value belongs to `normrow-002`
(section 3.4).

The question was which change flips the ratio, because the answer decides what
an upstream filing may claim: a property of the tt-metal rms kernel, or a
caller-side parameter that changed in this tree.

This document answers it with a measurement: the flip is the ttnn rms_norm
compute-kernel configuration, and the two recorded sessions differ because they
ran two different tt-metal substrates, whose default rms_norm precision
configuration differs. No vllm.cpp caller parameter, harness change, row, or
checkpoint is involved.

## 2. What the measurement rules out

### 2.1 The harness micro is numerically unchanged

The micro case is `tests/vllm/models/test_kolibri1_tt_b2ii.cpp`, `TEST_CASE("SCRATCH dbg rmsnorm micro")`.
The commit that added the replay arm is 5ef353c31 (2026-10-17 09:36). The diff
from that commit to HEAD touches the micro case in three places, and none of
them changes a number:

```diff
-  // ITEM-2 replay arm (the rms_norm input-dependence diagnostic): set
-  // VT_TT_RMSNORM_MICRO_INPUT=<bin> to a row captured by VT_TT_NORM_DUMP_DIR
+  // ITEM-2 replay arm (not for landing): VT_TT_RMSNORM_MICRO_INPUT=<bin>
+  // loads a REAL in-model row captured by VT_TT_NORM_DUMP_DIR in
-  auto tobf = [](float f) {
+  auto tobf2 = [](float f) {
-  FILE* of = std::fopen("/tmp/normdump/tt-out.bin", "wb");
```

The first hunk rewrites a comment. The second renames the lambda that the
random-input arm uses; the replay arm keeps its own lambda and its own
conversion, so no value changes. The third hunk adds a dump of the device
output. The rest of the diff belongs to two other test cases (the band gate
and the chunked-prefill adjudication), not to the micro.

The replay arm reads the captured file, rounds the row and the gamma to bf16,
builds a `[1, d]` bf16 input, a `[1, d]` bf16 gamma, calls
`vt::RmsNorm(..., vt::RmsNormArgs{1e-6f, false}, ...)` and compares the
device output against a host-double oracle with eps = 1e-6. The residual arm
runs with an all-zero residual, so the two arms of the micro read the same.

### 2.2 The vllm.cpp call site is unchanged

`src/vt/tenstorrent/tenstorrent_ops.cpp` has no diff between 5ef353c31 and
HEAD. The device arm still reads:

```cpp
ttnn::Tensor wf32 = args.gemma ? dev_w
                               : ttnn::typecast(dev_w, ttnn::DataType::FLOAT32);
ttnn::Tensor norm_f32 = NormalizeDevF32Tile(std::move(to_norm), rows, d);
...
dev_y = ttnn::rms_norm(norm_f32, args.eps, wf32);
```

The call passes an f32 TILE, an f32 TILE gamma and eps, and no compute-kernel
configuration. The ttnn default therefore decides the numerics.

### 2.3 The rows, the scale grid and the checkpoint

The three rows are the same bytes in both replays. Their sha256 sums:

| file | sha256 |
|---|---|
| `normrow-000.bin` | `c627768a78df4aac07170c665605c2c0bbbc47938edd91506247480e8b858564` |
| `normrow-001.bin` | `0dbb3b3c270831d6266ec31e0d873a22f7b225b78cc3fc7299741b9038b904cf` |
| `normrow-004.bin` | `418e4bf2ab0fa4fa12ae4ee2734d1bdc67c25cb1592c93764c40c15e70256f78` |

The capture happened on 2026-10-17 08:35 and the files still carry that mtime.
The captured file holds three blocks: the f32 tile the device consumed, the
input row read back as f32, and the staged gamma. The micro uses the first row
of the input block and the first `d` values of the gamma block, so the scale
grid is the captured one in both replays.

The micro loads no checkpoint: it replays one captured row standalone, so the
model weights and the checkpoint state cannot explain the difference.

### 2.4 The vllm.cpp binaries are locked to their own substrate

A binary built against one tt-metal does not run on the other. The cross-over
fails at load time, not at the kernel:

```
/tmp/build-p150-base/tests/test_kolibri1_tt_b2ii: symbol lookup error: undefined symbol:
_ZN4ttnn3sumERKNS_6TensorERKSt8optionalISt7variantIJilN4ttsl11SmallVectorIiLm8EEEEEEbRKS3_IN2tt8tt_metal12MemoryConfigEERKS3_INS_19ComputeKernelConfigEEfbRKS3_INSD_12CoreRangeSetEE
```

So the two recorded sessions cannot have run the same binary against different
substrates by accident, and the substrate must be named in any filing.

## 3. The bisect

### 3.1 The two substrates

| substrate | tt-metal source | install | rms_norm default |
|---|---|---|---|
| pinned evidence | `/tmp/tt-metal-umdtrial` @ 5b195891962 plus two local working-tree edits | `/tmp/umdtrial-install` (ttnn re-installed 2026-10-10 17:36) | `approx_mode = false`, `fp32_acc = true` |
| upstream head | `/tmp/tt-metal-new` @ d1be4cf5b4f (on 2500c1d8f63) plus two local build edits | `/tmp/umdtrial-new-install` (2026-10-20 00:09) | `approx_mode = true`, `fp32_acc = false` |

The pinned fork's local edit is in
`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp`:

```diff
-    bool approx_mode = true;
-    bool fp32_acc = false;
+    bool approx_mode = false;
+    // vllm.cpp pin-forward (rms reduction precision): the RMSNorm reduction
+    // (sum of squares, mean, rsqrt) must accumulate in f32. ...
+    bool fp32_acc = true;
```

and a second local edit in
`ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp`
gives the 1/W scaler tile a Float32 format when `fp32_dest_acc_en` is set.
Upstream main at 2500c1d8f63 still carries `approx_mode = true` and
`fp32_acc = false`.

### 3.2 The measurement

Two harnesses were run under the file mutex with the recipe of the original
sessions (device reset, tt-metal cache and `/dev/shm` cleared, the row replayed
through the micro):

| binary | substrate | row 000 | row 004 | row 001 |
|---|---|---|---|---|
| `/tmp/build-p150-base/tests/test_kolibri1_tt_b2ii` (2026-10-17 20:35) | `/tmp/umdtrial-install` | **1.0072** | **1.00771** | **0.995782** |
| `/tmp/vllm-newsubstrate/build/tests/test_kolibri1_tt_b2ii` (2026-10-20 00:22) | `/tmp/umdtrial-new-install` | **0.980947** | **1.0033** | **0.996982** |

Both readings reproduce exactly, so both records are confirmed as measurements
and not as transcription errors. The pinned reading is stable: two repeats
read 1.0072 and 0.991758 (tt/cpu) both times.

The pinned binary was built on 2026-10-17 20:35 from a source tree that has
since been deleted. Its micro is the one of 5ef353c31, because the micro's
replay code does not change between 5ef353c31 and HEAD, so every commit in
that window carries the same micro.

The two vllm.cpp trees are equivalent for this path. The upstream tree is
61d333e52 plus four API-drift shims (an include path, a forward declaration
with new default arguments, one extra default argument at a paged-attention
call, and a moved spec header). No shim touches the rms path, and
`src/vt/tenstorrent/` has no diff between 61d333e52 and HEAD.

### 3.3 The single variable: the rms_norm compute config

A standalone ttnn program (one translation unit, linked against each
substrate) replays the same rows and selects the compute-kernel configuration
explicitly. Its default arm reproduces the harness numbers exactly
(1.007197 on the pinned substrate, 0.980947 on the upstream substrate, and the
same 676 differing elements on row 000), so it measures the same path.

`ratio_bf16` is the sum of the device bf16 output divided by the sum of the
host-double bf16 oracle, the quantity the harness prints as `tt/host2`:

| mode | `approx_mode` | `fp32_acc` | pin row 000 | pin row 004 | pin row 001 |
|---|---|---|---|---|---|
| 0 (default) | substrate default | substrate default | 1.007197 | 1.007708 | 0.995782 |
| 1 | false | true | 1.007197 | 1.007708 | 0.995782 |
| 2 | true | false | 0.980947 | 1.003302 | 0.996982 |
| 3 | false | false | 0.980947 | 1.003302 | 0.996982 |
| 4 | true | true | 0.995718 | 1.009895 | 0.995782 |

The same five modes on the upstream substrate give byte-identical numbers:

| mode | upstream row 000 | upstream row 004 | upstream row 001 |
|---|---|---|---|
| 0 (default) | 0.980947 | 1.003302 | 0.996982 |
| 1 | 1.007197 | 1.007708 | 0.995782 |
| 2 | 0.980947 | 1.003302 | 0.996982 |
| 3 | 0.980947 | 1.003302 | 0.996982 |
| 4 | 0.995718 | 1.009895 | 0.995782 |

Two results follow.

1. The compute-kernel configuration alone decides the number. `fp32_acc = false`
   reads 0.980947 on row 000 on both substrates, whatever `approx_mode` is.
   `fp32_acc = true` reads 1.007197 with `approx_mode = false` and 0.995718
   with `approx_mode = true`.
2. The tt-metal version does not matter for these rows. The upstream rework
   (`rsqrt_tile<LEGACY_RSQRT>` to `ckl::Rsqrt<Approx::Exact>` in a new
   `eltwise_chain`) changed no measured number: mode 1 on upstream reads
   1.007197, the same value the pinned fork reads in its own default.

### 3.4 The whole row set

Every row moves with the configuration, and the sign of the deviation is not
predictable from the row magnitude:

| row | shape | max&#124;x&#124; | pinned default | upstream default |
|---|---|---|---|---|
| `normrow-000` | 1x2560 | 3.54688 | 1.007197 | 0.980947 |
| `normrow-001` | 48x128 | 0.949219 | 0.995782 | 0.996982 |
| `normrow-002` | 4x128 | 0.6875 | 0.998398 | 1.006526 |
| `normrow-003` | 1x2560 | 0.169922 | 1.003225 | 0.996909 |
| `normrow-004` | 1x2560 | 5.5 | 1.007708 | 1.003302 |
| `normrow-005` | 1x2560 | 0.503906 | 0.995723 | 0.989580 |
| `normrow-007` | 48x128 | 1.44531 | 0.989715 | 1.001155 |

The record in ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF quotes 0.9984 for the
d=128 rows on the pin. That value is `normrow-002`, not `normrow-001`.

## 4. The reconstruction

| when | what | evidence |
|---|---|---|
| 2026-10-09 17:23 | the trial substrate is installed from `/tmp/tt-metal-umdtrial` (unpatched default: `approx_mode = true`, `fp32_acc = false`) | `libtt_metal.so` mtime |
| 2026-10-10 17:34 | the rms_norm precision edit lands in `/tmp/tt-metal-umdtrial` | `rmsnorm.cpp` mtime |
| 2026-10-10 17:36 | the substrate is re-installed with the edit | `_ttnncpp.so` mtime |
| 2026-10-17 08:35 | the rows are captured by `VT_TT_NORM_DUMP_DIR` | the row files' mtime |
| 2026-10-17 20:35 | the pinned replay binary is built | the binary's mtime |
| 2026-10-18 | the pinned replay reads 1.0072 / 1.0077 / 0.9984 | ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF |
| 2026-10-19 | tt-metal upstream 2500c1d8f63 is built at `/tmp/tt-metal-new` | the tree's history |
| 2026-10-20 00:09 | the upstream substrate is installed | `/tmp/umdtrial-new-install` mtime |
| 2026-10-20 00:22 | the upstream replay binary is built | the binary's mtime |
| 2026-10-20 | the upstream replay reads 0.9809 / 1.0033 / 0.9970 | ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF |
| 2026-10-21 08:46 | this re-measurement reproduces all six numbers | this document |

The 0.9809 family is the regime of a substrate without the local precision
edit, because `fp32_acc = false` reads 0.980947 on the pinned substrate too
(mode 2 and mode 3 above). The 1.0072 family is the regime of the patched
substrate, which is what the capture session and every replay since 2026-10-10
used.

The same install path therefore read 0.9809 between 2026-10-09 17:23 and
2026-10-10 17:36, before the precision edit was re-installed into it. A replay
on that window's substrate would read the upstream numbers on our own path,
which is why a "same path, two readings" report is possible without any
vllm.cpp change.

## 5. What an upstream filing may claim

The evidence supports the first claim, with two corrections.

Supported: the deviation is a property of the tt-metal rms_norm kernel, and
it is not a fixed bias. It is a function of the input row and of the
compute-kernel configuration the kernel runs under, and the caller in this
tree does not choose that configuration.

Refuted: the idea that the deviation depends on a caller-side parameter that
changed in this tree. The vllm.cpp rms path is byte-identical across the two
sessions, and the call passes no compute configuration. The parameter that
changed is tt-metal's own default, which this fork patches locally and which
upstream main still ships as `approx_mode = true`, `fp32_acc = false`.

Corrected: the record in ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF attributes
the 0.9809 reading to the upstream rsqrt rework. The measurement does not
support that. Upstream main and the pinned fork produce the same number for
the same configuration, so the 1793 upstream commits changed nothing in the
rms path on these rows, and the upstream rsqrt rework neither caused nor
fixed the difference.

A filing that is honest in both directions can say:

> `ttnn::rms_norm` on a real in-model activation row deviates from a host-f64
> oracle by -1.9% in the summed output (0.9809 on a 1x2560 row, every element
> differing) when it runs with the default compute configuration
> (`fp32_dest_acc_en = false`, `approx_mode = true`), and by +0.7% (1.0072)
> when the same kernel runs with `fp32_dest_acc_en = true` and
> `approx_mode = false`. The deviation is input-dependent in both regimes and
> changes sign between rows, so it is not a constant factor. The fused
> kernel's reduce and round composition is the defect; the reduction precision
> changes its size but does not remove it.

## 6. Uncertainty

- The bisect measures the compute configuration on seven captured rows from
  one model on one device. It does not prove that every tt-metal rms path
  behaves the same way.
- The pinned fork's local precision edit is not in the recorded pin source
  tree (`~/Sources/tt/tt-metal-pin` carries `approx_mode = true`,
  `fp32_acc = false`) and not in the recorded patch series
  (`~/Sources/tt/tt-metal-patches/0001`..`0004`). The installed evidence
  substrate carries it, and this document measures the installed substrate
  directly, so the numbers here stand; the pin record does not yet reproduce
  them. That gap is filed as ISSUE-LOCAL-01M5EM67GWF0D79SSY3NHZRKQA.
- The upstream build could not use the official SFPI 7.88.0 tarball (the
  `rvtt_live` ICE recorded in ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF) and used
  a local SFPI 7.83.0 build instead. A different SFPI could in principle
  change a kernel result, so the upstream numbers carry that caveat. The
  pinned numbers do not: the pinned substrate is the one the whole P150
  evidence chain used.
- The per-element structure of the deviation was not re-derived here beyond
  the differing-element counts (676 of 2560 on row 000 in the 1.0072 regime,
  2560 of 2560 in the 0.9809 regime). The claim that the deviation is not a
  constant per-row factor is consistent with these counts but is not the
  subject of this document.

## 7. Recipe

Device reset and run, under the file mutex, with the pinned substrate:

```sh
exec 9>$HOME/gpu.lock
flock -x 9
~/Sources/tt/.venv/bin/tt-smi -r 0 -l 120
sleep 20
rm -rf ~/.cache/tt-metal-cache/* /dev/shm/TT_UMD_LOCK.* \
       /dev/shm/tt_device_*_memory /dev/shm/sm_segment.thalia.*
env TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium \
    LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium \
    VT_TT_RMSNORM_MICRO_INPUT=/tmp/normdump/normrow-000.bin \
    taskset -c 32-63 /tmp/build-p150-base/tests/test_kolibri1_tt_b2ii \
    --test-case='SCRATCH dbg rmsnorm micro'
```

The upstream substrate uses `/tmp/umdtrial-new-install` in both environment
variables and the binary `/tmp/vllm-newsubstrate/build/tests/test_kolibri1_tt_b2ii`.
The standalone probe (`/tmp/rms-ttprobe/probe.cpp`, scratch, not committed)
takes the row file and a mode number and prints the same ratio.
