# SFPI 7.89.0 compiles the Blackhole layernorm specialization (2026-10-21)

Question: does the official SFPI 7.89.0 aarch64 toolchain still hit the
`rvtt_live` ICE that SFPI 7.88.0 hits while compiling the Blackhole layernorm
compute-kernel specialization?

Answer: no. The 7.89.0 compiler compiles and LTO-links the Blackhole
layernorm specialization, the whole tt-metal tree builds clean against
7.89.0, and the device rms micro runs with the same numbers it produces
under 7.88.0. An upstream report is therefore a "fixed in 7.89" confirmation
and not a new defect.

The measurement also corrects the premise. The ICE reproduces with the
locally built 7.88.0 toolchain, not with the official 7.88.0 release
tarball. The official 7.88.0 tarball compiles the same specialization in the
same harness.

This is the re-check that `ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF` names as
its trigger: "SFPI >= 7.89 fixing rvtt_live".

## 1. Artifacts

| Item | Value |
|---|---|
| Toolchain under test | `sfpi_7.89.0_aarch64_debian.txz`, 73,534,308 bytes |
| Toolchain sha256 | `78b4460a697966393f8a0ccf18b071ee610b768650ffbecadc4a9e066b0fb9a7` |
| Release asset digest | `sha256:78b4460a...`, identical to the local bytes, from the `tenstorrent/sfpi` 7.89.0 release |
| Version file | release asset `sfpi-version`, sha256 `ce61bdae55539f76833dbca742766a48503e308f3b6b52b0a4771e262de84e5c` |
| Control toolchain, official tarball | `sfpi_7.88.0_aarch64_debian.txz`, sha256 `5fa182683ad068c687e7718c02675371cfa11059ed4ba277d0cdca554f16d65a` |
| Control toolchain, locally built 7.88.0 | `riscv-tt-elf-g++ (tenstorrent/sfpi-DIY:7.88.0) 15.1.0`, kept by the probe at `/tmp/umdtrial-new-install/libexec/tt-metalium/runtime/sfpi-7.88.0` |
| Compiler under test | `riscv-tt-elf-g++ (tenstorrent/sfpi:7.89.0[1050]) 15.1.0` |
| Control compiler | `riscv-tt-elf-g++ (tenstorrent/sfpi:7.88.0[1039]) 15.1.0` |
| tt-metal tree | `/tmp/tt-metal-new` at `origin/main` `2500c1d8f63` |
| Build tree additions | `8f95986a422` (aarch64/Gentoo build accommodations), `d1be4cf5b4f` (host-side layernorm argument refresh) |
| New build directory | `/tmp/tt-metal-new/build_Release_789` |
| New install prefix | `/tmp/umdtrial-789-install` |
| Separate SFPI prefixes | `/tmp/sfpi-7.88.0/sfpi`, `/tmp/sfpi-7.89.0/sfpi` |
| Device probe tree | `/tmp/vllm-newsubstrate`, binary `build/tests/test_kolibri1_tt_b2ii` |
| Device probe harness | `/tmp/micro-new.sh` |

The pinned evidence substrate `/tmp/umdtrial-install` with
`/tmp/tt-metal-umdtrial` was not modified. The probe build
`/tmp/tt-metal-new/build_Release` was not modified, and the probe install
`/tmp/umdtrial-new-install` was modified only in the toolchain under
`libexec/tt-metalium/runtime/sfpi`, which the three arms of section 4 swap
and restore.

The SFPI C++ headers differ between the two releases. `sfpi.h`,
`sfpi_classes.h`, `sfpi_funcs.h` and `tensix_builtins.def` all differ, so
7.89.0 is not a rebuild of the 7.88.0 sources.

## 2. The tt-metal build against SFPI 7.89.0

The tree selects the toolchain through `tt_metal/sfpi-version`, which the
`sync` of the 7.89.0 release asset replaced with the upstream 7.89.0 file.
The official aarch64 tarball was placed next to the 7.88.0 one under the name
the Gentoo distribution expects, so the build unpacks it instead of building
SFPI from source:

```sh
cp /tmp/sfpi789.txz /tmp/tt-metal-new/runtime/sfpi_7.89.0_aarch64_gentoo.txz
cp /tmp/sfpi789-assets/sfpi-version /tmp/tt-metal-new/tt_metal/sfpi-version
rm -rf /tmp/tt-metal-new/runtime/sfpi
```

The configure step and the build:

```sh
cmake -S /tmp/tt-metal-new -B /tmp/tt-metal-new/build_Release_789 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/tmp/umdtrial-789-install \
  -DCMAKE_CXX_COMPILER=/usr/lib/llvm/20/bin/clang++-20 \
  -DCPM_SOURCE_CACHE=/tmp/tt-metal-new/.cpmcache \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPython3_INCLUDE_DIR=/usr/include/python3.13 \
  -DPython3_LIBRARY=/usr/lib64/libpython3.13.so \
  -DTT_UMD_BUILD_TOOLS=OFF -DTT_UMD_BUILD_PYTHON=OFF -DTT_UMD_BUILD_ALL=OFF \
  -DTT_UMD_BUILD_TESTS=OFF -DTT_UMD_BUILD_EXAMPLES=OFF \
  -DTT_METAL_BUILD_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_TT_TRAIN=OFF \
  -DBUILD_PROGRAMMING_EXAMPLES=OFF -DTT_ENABLE_LTO=OFF -DENABLE_FAKE_KERNELS_TARGET=OFF
ninja -C /tmp/tt-metal-new/build_Release_789 -j 8
```

The configure log names the toolchain that the build uses:

```text
-- No downloadable SFPI tarball for aarch64 gentoo
-- Locally-built sfpi_7.89.0_aarch64_gentoo.txz present
-- Unpacking locally-built sfpi_7.89.0_aarch64_gentoo.txz
-- Using SFPI compiler: /tmp/tt-metal-new/runtime/sfpi/compiler/bin/riscv-tt-elf-g++ (tenstorrent/sfpi:7.89.0[1050]) 15.1.0
```

Result: the build finished with exit code 0 after 10 min 42.7 s of wall time
and 86 min 20.8 s of user time, with 1437 of 1437 targets built. The last
target, "Pre-compiling firmware binaries", compiles the device firmware with
the SFPI compiler, so the 7.89.0 toolchain compiled device code as part of
this build. The firmware bundle landed in
`/tmp/tt-metal-new/tt_metal/pre-compiled/4705343793634249657/`. No compiler
error and no ICE appeared in the build log.

## 3. The layernorm specialization compile, 7.88.0 against 7.89.0

The layernorm specialization is a device kernel that the JIT compiles at
runtime, so the host build in section 2 does not reach it. This section
compiles it without a device.

The harness drives the tt-metal JIT directly, with a mock Blackhole target so
that no device is used. It reuses the internal build API of tt-metal:
`BuildEnvManager`, `JitBuildOptions`, `jit_build_genfiles_triscs_src` and
`jit_build_subset`, over the Blackhole descriptors
`tt_metal/core_descriptors/blackhole_140_arch.yaml` and
`tt_metal/soc_descriptors/blackhole_140_arch.yaml`. The kernel is the
Metal 2.0 layernorm specialization of the ttnn rms_norm op, built with the
same bindings that op uses: `is_metal2_kernel` true, runtime argument names
`{NCHt}`, the dataflow-buffer accessors `in`, `eps`, `scaler`, `ex2`,
`xmm2`, `ex2pe`, `gamma` and `fusion`, the named compile-time arguments
`Wt`, `block_size`, `do_gamma`, `do_beta`, `fp32_dest_acc_en`, `W`,
`tile_width` and `float32_reduction`, and the defines `RMSNORM` and
`FUSE_GAMMA`. The kernel source is
`ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp`.

Each arm has its own runtime root, its own JIT cache, its own precompiled
firmware and its own build of the four Blackhole hardware objects
(`substitutes.o`, `tdma_xmov.o`, `noc.o`, `tmu-crt0.o`), so the only
difference between the arms is the SFPI compiler:

| Arm | Runtime root | SFPI compiler | Hardware objects |
|---|---|---|---|
| Control | `/tmp/sfpi-ab/root788` | 7.88.0[1039] | built with 7.88.0 |
| Test | `/tmp/sfpi-ab/root789` | 7.89.0[1050] | built with 7.89.0 |

The compile command that the JIT runs for the specialization is the same in
both arms apart from the compiler path and the output paths:

```text
riscv-tt-elf-g++ -O3 -std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval \
  -ftt-no-dyninit -flto=1 -flto-partition=one -ffast-math \
  -fno-finite-math-only -fsigned-zeros -fno-associative-math -fno-exceptions \
  -fno-rtti -fno-use-cxa-atexit -g -MMD -Wall -Werror \
  --param=min-pagesize=0 -mcpu=tt-bh-tensix -include <pch.h> \
  -DUCK_CHLKC_MATH -DCOMPILE_FOR_TRISC=1 -DARCH_BLACKHOLE ... \
  -c -o <cache>/kernels/probe_ln/0/trisc1/trisck.<hash>.o \
  <root>/tt_metal/hw/firmware/src/tt-1xx/trisck.cc
```

Result:

| Arm | Compile commands run | `rvtt_live` or `verify_ssa` hits | Outcome |
|---|---|---|---|
| 7.88.0 | 36 | 0 | compile and LTO link finished, `OFFLINE COMPILE OK` |
| 7.89.0 | 36 | 0 | compile and LTO link finished, `OFFLINE COMPILE OK` |

Logs: `/tmp/sfpi-ab/final_ln788.log` and `/tmp/sfpi-ab/final_ln789.log`. The
firmware for the link step was produced with the same arm compiler by
`build_Release_789/tools/precompile_fw`, which also completed with exit code 0
and no ICE in both arms.

## 4. The device rms micro, three toolchain arms

The device rms micro of the vllm.cpp probe reaches the specialization. The
micro replays the captured in-model rows
`/tmp/normdump/normrow-000.bin`, `normrow-004.bin` and `normrow-001.bin`
through the tt-metal rms path, and the JIT compiles the layernorm
specialization for Blackhole with the toolchain under
`/tmp/umdtrial-new-install/libexec/tt-metalium/runtime/sfpi`.

The probe is the vllm.cpp tree at `/tmp/vllm-newsubstrate` with the
`test_kolibri1_tt_b2ii` binary and the harness `/tmp/micro-new.sh`. The
harness takes the device lock, resets the device, clears the JIT cache, and
runs the micro once per captured row. Only the toolchain under
`runtime/sfpi` changes between the arms. The compiler in each arm reports
its own version, so the arm identity is not a claim:

| Arm | Compiler under `runtime/sfpi` | `rvtt_live` hits | Micro result |
|---|---|---|---|
| 1 | `riscv-tt-elf-g++ (tenstorrent/sfpi-DIY:7.88.0) 15.1.0`, locally built | 3 ICE sites | FAILURE |
| 2 | `riscv-tt-elf-g++ (tenstorrent/sfpi:7.88.0[1039]) 15.1.0`, official tarball | 0 | SUCCESS |
| 3 | `riscv-tt-elf-g++ (tenstorrent/sfpi:7.89.0[1050]) 15.1.0`, official tarball | 0 | SUCCESS |

The ICE of arm 1 is the reported failure, one site per TRISC:

```text
/tmp/umdtrial-new-install/libexec/tt-metalium/tt_metal/hw/firmware/src/tt-1xx/trisck.cc:65:10: error: missing definition
   65 | uint32_t _start() {
      |          ^
for SSA_NAME: .MEM_220 in statement:
# .MEM_201 = VDEF <.MEM_220>
__builtin_rvtt_sfppushc (0);
during GIMPLE pass: rvtt_live
```

Arms 2 and 3 produce the same numbers, so the toolchain does not move the
kernel numerics:

| Row | Arm 2, official 7.88.0 | Arm 3, official 7.89.0 |
|---|---|---|
| normrow-000, 1x2560 | `tt/host2=0.980947` | `tt/host2=0.980947` |
| normrow-004, 1x2560 | `tt/host2=1.0033` | `tt/host2=1.0033` |
| normrow-001, 48x128 | `tt/host2=0.996982` | `tt/host2=0.996982` |

Logs: `/tmp/evidence-788diy/`, `/tmp/evidence-788off/` and
`/tmp/evidence-789off/`, one file per captured row.

## 5. What this measurement establishes

- SFPI 7.89.0 does not hit the `rvtt_live` ICE. The Blackhole layernorm
  specialization compiles and links with 7.89.0, the whole tt-metal tree
  builds clean against 7.89.0, and the device rms micro runs with the same
  numbers it produces under 7.88.0.
- The ICE reproduces with the locally built 7.88.0 toolchain and not with
  the official 7.88.0 release tarball, in the same tree, with the same
  kernel, and with the same flags. The defect is therefore a property of
  that local build, and the record that calls the official tarball the
  failing one names the wrong toolchain.
- The device-free harness of section 3 compiles the same specialization with
  both 7.88.0 and 7.89.0, so it does not reproduce the ICE on its own. The
  device micro of section 4 does.
- The 7.88.0 failure is documented in the project record. The session memory
  `ttmetal-upstream-verdict.md` of 2026-10-20 states that the 7.89.0 tarball
  fixes the `rvtt_live` and `verify_ssa` ICE, that the tt-metal tree builds
  clean, and that the layernorm specialization compiles and runs. This
  measurement agrees with that verdict.
- No different failure appeared in the 7.89.0 build or in the 7.89.0 arm of
  the specialization compile. The only failure the harness produced was in
  both arms, and it was a link error that the harness caused: the kernel link
  needs the weakened firmware ELF, and the harness had none until section 3
  built the firmware.

## 6. Residue

- Probe tree `/tmp/tt-metal-new`: `tt_metal/sfpi-version` now names 7.89.0,
  `runtime/sfpi` holds the 7.89.0 toolchain, and
  `runtime/sfpi_7.89.0_aarch64_gentoo.txz` holds the official tarball. The
  7.88.0 tarball and the 7.89.0 install prefix stay in place.
- Probe build `/tmp/tt-metal-new/build_Release_789` and probe install
  `/tmp/umdtrial-789-install` are new.
- Harness sources and logs stay under `/tmp/sfpi-ab/`.
- The device probe tree `/tmp/vllm-newsubstrate` and the new substrate
  install `/tmp/umdtrial-new-install` are unchanged apart from the toolchain
  swap of section 4. The runtime directory holds the official 7.89.0 toolchain
  under `runtime/sfpi` and the locally built 7.88.0 under
  `runtime/sfpi-7.88.0`, which is how the probe found them.
- The pinned evidence substrate `/tmp/umdtrial-install` with
  `/tmp/tt-metal-umdtrial` was not touched by this measurement. A locally
  built 7.88.0 against the pinned tree is not part of this record.
