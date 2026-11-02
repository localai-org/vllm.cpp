# Legacy prefill shared-memory repair

Issue: `ISSUE-LOCAL-01M63NES5TYK3E3WH5B50XVAPP`.
Spec-first commit: `44c22ada7`.
Base implementation: `03b19115a83f1c257b630c1e3bece7361ce08e95`.
Date: 29 October 2026.

## Observed failure and change

A public-loader synthetic one-layer Qwen3.5 fixture with head dimension 256
loads and generates on CPU. P620 CUDA completion with prompt length 16 fails
because portable prefill requests 65536 bytes of shared memory. The queried
P620 per-block limit is 49152 bytes. `red-run.log` records the intended failure.

The existing portable kernel receives its key-tile width at runtime. On a valid
pre-sm_80 device, halve that width until both F32 K/V staging buffers fit the
queried ceiling. P620 head256 uses 16 keys and 32768 bytes. Head64 and head128
keep 32 keys. The order of each query's online-softmax keys stays unchanged.
The sm_80-and-newer tile choice and existing shared-memory guard remain intact.
This is a local tile adaptation. The pinned secondary dispatch source is
llama.cpp `ggml/src/ggml-cuda/fattn.cu:757-775` at
`62318cd57dc63900ecb153272c0b10f78550855d`.

## Focused gate

The test now enters public load and completion for head dimensions 64, 128, and
256, each with prompt lengths 1, 16, and 37 and four generated tokens. All nine
CPU/CUDA token comparisons pass. Every callback captures finite full logits.
The CUDA path records no CPU reference fallback. This is a synthetic regression,
not a published-checkpoint or performance gate.

`python3 /tmp/smem-driver.py red` exited 1 at head256 prompt16 before the fix.
`python3 /tmp/smem-driver.py green /tmp/smem-repair/paged-green.o` exited 0.
The driver compiles the actual production translation unit from the configured
compile database, substitutes it in a scratch archive, and uses the existing
public-CAPI test link command. It compiles fresh production platform and legacy
registrar objects. The original build and source tree are not modified.
`reproduce.py` retains the scratch command construction.

The repaired translation unit compiles with CUDA 12.4, GCC 13, C++20, Release,
`-O3 -DNDEBUG`, warnings as errors, and sm_60, sm_61, sm_70. `cuobjdump --list-elf
/tmp/smem-repair/paged-green.o` lists all three cubins. No Volta or P100 runtime
claim follows from compilation. All GPU runs use `flock /home/user/gpu.lock`.

A supplementary scratch consumer includes the old fixture in a renamed namespace
and compares its default GGUF bytes with the parameterized fixture. The bytes
are equal and all nine completion cases pass, exit 0. See
`fixture-compat-green-run.log`. The old fixture is unchanged at the base head.
Head192 cannot divide this fixture's model width of 256, so it is not exercised.

`git diff --check`, `python3 scripts/check-agent-record.py`, and
`cmake -P cmake/CudaArchFeaturesTest.cmake` each exit 0.

## Mutation and restoration

In a scratch source copy, replace the new legacy capability branch with an
always-false condition. Compile that actual production translation unit for
sm_61. `python3 /tmp/smem-driver.py mutation /tmp/smem-repair/paged-mut.o`
exits 1 at head256 prompt16 with the same 65536-byte request and 49152-byte
limit. See `mutation-run.log`. The production source is never mutated. The
restored green object passes all nine workloads. An initial harness copied a
reviewer's mutated priority object and produced an unrelated attention-selection
refusal. That run is excluded. Both platform and registrar were then rebuilt
from the immutable production source before all retained observations.

## Pending gates

Full repository preflight was started before editing. Its log is
`/tmp/smem-preflight.log`; it is incomplete, not green. The parent already tracks
unmodified baseline failures and row-record repairs. The full Release library
build, post-integration focused suite, independent review, and operator gate are
still required. Real checkpoints, performance axes, P100 runtime, and Volta
hardware validation remain pending under the owning spec.
