# turboderp-org `exllamav3` — the EXL3 trellis format, and the DeepSeek-V4 support that reads it

EXL3 is a variant of **QTIP**: a procedural codebook encoding high-dimensional
vectors into tail-biting trellis structures, deviating from QTIP in how tensors
are regularized and packed (`doc/exl3.md:3`, `README.md:6,171`). vLLM defines no
such format — `layers/quantization/` at the parity pin
`5559679229bc961848b121ccdeaa8fa5d79bec98` registers no `exl3`, `exllamav3` or
trellis method — so `MODEL-DSV4-EXL3` is the fallback case the rule admits, and
this file is the pin it requires.

**Provenance of that vLLM negative: MEASURED, 2026-08-25.** It was first
established by the row's W1 spike and recorded in
[`../specs/model-dsv4-exl3.md`](../specs/model-dsv4-exl3.md), and it has since
been re-derived first-hand in the pinned checkout resolved from `.env`
(`VLLM_SOURCE`), at `HEAD = 5559679229bc`, which is the parity pin itself:

```sh
grep -rl -i 'exl3\|exllamav3\|trellis' --include='*.py' --include='*.cu' \
     --include='*.cuh' --include='*.h' .      # -> 0 files
ls vllm/model_executor/layers/quantization/*.py
# auto_awq auto_gptq awq_triton base_config bitsandbytes experts_int8
# fbgemm_fp8 fp8 fp_quant humming input_quant_fp8 kv_cache modelopt
# moe_wna16 mxfp4 qutlass_utils torchao  -- no EXL3, no trellis method
```

Zero matches across the whole tree, and no EXL3 entry among the registered
quantization methods. The earlier draft of this file said the negative could
not be re-derived here because no vLLM checkout was on the host; that was wrong
— the pinned checkout is the one `.env` names, outside the writing agent's
worktree. The claim now rests on a measurement rather than on a citation, which
matters because it is the single premise the whole registration stands on: if
vLLM implemented EXL3, this oracle would be inadmissible under the primary
rule.

**Why the fallback is DeepSeek-V4-shaped rather than only format-shaped.** The
pinned HEAD registers `DeepseekV4Model` as a first-class architecture
(`exllamav3/architecture/architectures.py:8,70`) alongside a DSpark drafter
(`deepseek_v4.py:10,30` -> `deepseek_v4_mtp.py`, 372 lines), a 1744-line
`exllamav3/modules/dsv4.py`, and three CUDA kernels with tests behind them:
`exllamav3/exllamav3_ext/{dsv4_compress.cu,dsa_topk.cu,hc_mix.cu}` and
`tests/test_dsv4_{cached,compress_kernel,state}.py`. Note the kernel path is
nested — `exllamav3/exllamav3_ext/`, not a top-level `exllamav3_ext/`.

**It keys on the tensor names our checkpoint actually ships.** `deepseek_v4.py`
builds `layers.{idx}` (`:134`), `{key}.ffn` (`:167`) and
`experts.{expert_idx}.{w1,w2,w3}` (`:172-174`), which composes to
`layers.0.ffn.experts.0.w1` — the name
`.agents/specs/model-dsv4-exl3.md` records reading out of
`exl3-layer-000-tp4-rank1.safetensors` as `layers.0.ffn.experts.0.w1.rank1`. The
staged `config.json` declares `architectures = ["DeepseekV4ForCausalLM"]`,
`model_type = "deepseek_v4"`, and
`quantization_config = {quant_method: "exl3", bits: 3.0, codebook: "mcg",
version: "rank-sliced-deepseek-v4-v1"}`. So the artifact and this upstream are
the same key scheme, not two conventions that happen to overlap.

**The checkpoint was quantized by an OLDER revision of this same repository.**
Its `config.json` carries
`hybrid_tr3_tail.exllamav3_revision = 787d1582267117d6ee83c90014f03b525b14754f`
(and `hybrid_tr3_tail.source_revision = 9e165c30e2704aec5d9d593cce3eebd58bbef1cb`),
read 2026-08-25 from the NAS-staged
`/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3/config.json`. One limit
remains on that fact: the key is scoped to `hybrid_tr3_tail` rather than declared
for the whole artifact.

**The ancestry that the first draft left open is now MEASURED, 2026-08-26.** That
draft could not place `787d1582` because its clone was `depth = 1`, and it said
so rather than reading absence as evidence. `git fetch --unshallow` (1472
commits, 12 MB) settles it:

```sh
git merge-base --is-ancestor 787d1582267117d6ee83c90014f03b525b14754f HEAD  # -> 0
git log --oneline -1 787d1582        # -> 787d158 Bump to v1.2.1
git rev-list --count 787d1582..HEAD  # -> 161
```

So the revision that quantized the checkpoint IS an ancestor of the pin, it is
upstream's own v1.2.1 tag commit, and the pin is 161 commits ahead of it. Two
consequences follow from the same history, and both matter to whether this
oracle may speak for this artifact:

- Between that revision and the pin, exactly ONE commit touches the EXL3 storage
  path (`modules/quant/exl3.py`, `quant/codebook.cuh`, `quant/exl3_dq.cuh`,
  `quant/pack.cu`): `d33be5c` "Reconstruct: Fused reconstruct for large-M
  matmuls". That is a compute fusion, not a layout change, so the pin reads what
  v1.2.1 wrote.
- DeepSeek-V4 support is ABSENT at `787d1582`
  (`git cat-file -e 787d1582:exllamav3/architecture/deepseek_v4.py` fails; it
  arrives later, in `dd50dfe` "Add DeepseekV4ForCausalLM" and `7a9f3d0` "Add DSA
  and DSv4 Compressor module"). The checkpoint was therefore quantized by an
  exllamav3 that could not itself run the architecture.

## Scope, and what this oracle may not do

Use it for the EXL3 format and its kernels — trellis packing and the MCG
codebook, the `suh`/`svh` Hadamard-and-sign vectors, `had_r_128`, `exl3_gemm`,
the `m <= 8` GEMV arm, `exl3_moe` — and for DeepSeek-V4 behavior that follows
from reading a checkpoint stored in that format. `LinearEXL3.quant_type` is
`"exl3"` (`exllamav3/modules/quant/exl3.py:18`).

It is a mirror source for ONE quantization family and its execution, not a
design reference. Where vLLM defines behavior — attention, MLA, MoE routing,
sampling, scheduling, the serving path — vLLM wins, exactly as `AGENTS.md`
requires. "exllamav3 does it differently" is never on its own a reason to
diverge, and when vLLM registers an EXL3 method the row reconciles onto vLLM and
records the change in its spec.

## A prerequisite this oracle imposes on any run

**`tp_import_split` is a runtime IPC split, not a reader of pre-sliced files.**
The producer side is a `multiprocessing.shared_memory` arena
(`exllamav3/model/model_tp_shared.py:23,40`), and every importer takes its slice
by `consumer.recv(..., slice_dim, first, last)` out of that arena:
`TPTensorWrapper.tp_import_split` at `model_tp_shared.py:285-292`, and
`LinearEXL3.tp_import_split` at `exl3.py:285-329`, which slices `svh` on dim 0
and `trellis` on dim 1 in `first // 16 .. last // 16` units. Nothing in the tree
reads a `.rank{r}` filename: a `grep -rn 'rank{|\.rank\b|rank_%d|f"rank'` over
its Python returns only NCCL rank bookkeeping in
`exllamav3/model/model_tp_backend.py`.

Our artifact ships 43 layers x 4 `.rank{r}` shards on disk. Therefore a
**TP1-coalesced checkpoint is a prerequisite** before exllamav3 can load it on
one GB10 — which is what W1b's `LoadDeepseekV4Exl3` already produces — or the
run needs four GPUs that a single Spark does not have. This is a scheduling
fact, not a defect in either side.

**And coalescing to TP1 is not sufficient, because the KEY NAMES have to match
too.** The TP1 artifact that `MODEL-DSV4-EXL3`'s denominator run produced is not
loadable by this oracle: its coalescer writes `{prefix}.rank0.{field}`
(`image-patch/coalesce_rank_sliced_exl3.py`, the `output_key` assignment),
because SparkInfer's loader wants a rank segment even at TP1. exllamav3 wants
none. `Linear.is_exl3_storage` requires `{key}.trellis` together with
`{key}.suh|.su` and `{key}.svh|.sv` (`modules/linear.py:385-389`), and
`load_exl3` reads `key + ".trellis"` (`linear.py:406`), so a `.rank0.` segment
makes every EXL3 linear miss and the module falls through to the dense loader.
Anything that feeds this oracle has to emit exllamav3-native keys.

## Gateability

`gateable = no`, and 2026-08-26 turned that from an unmeasured field into a
measured one. [#1901](https://github.com/mudler/vllm.cpp/issues/1901) still owns
the half the fleet would not let anyone reach.

**AMD addendum, measured 2026-10-05.** The verdict above is scoped to the
aarch64 CUDA fleet and to source builds of the `2398c056` pin. On the host's
RX 7900 XTX (gfx1100) the prebuilt ROCm image
`exllamav3-rocm:git-679835b7-rocm10.0.0` demonstrably loads and serves an EXL3
checkpoint that no earlier image can: `orcarouter/OrcaSAQ-2-27B-EXL3-3.21bpw`,
whose `.trellis` tensors carry per-tensor mixed rates including 120 tensors at
the half-integer `K = 3.5` (56 uint16 words per tile). Two greedy 256-token
jobs under `~/gpu-coord` lease produced coherent ChatML output at ~24 tok/s
(token IDs archived at `.agents/specs/orca_golden_{0,1}.json` for
`BACKEND-ROCM`'s frac-rate token gate). This image is therefore the gateable
oracle for half-integer EXL3 rates on ROCm — the `2398c056` pin predates
`quant/frac.cu` and `dq8_half` entirely (first seen in `679835b7`), so it
cannot adjudicate that format. `gateable = no` above stands for the
CUDA-fleet source build; the ROCm image's reach ends at EXL3-format work on
gfx1100, which is exactly what `quant-exl3-frac-rates` needs.

**The verdict: the pin does not BUILD on this fleet, because exllamav3 at
`2398c056` is an x86_64 project and every device here is aarch64.** Measured
inside three `rc` leases on `orin:gpu0` (worker `rc-worker-lnvw6` and boot_id
`7b196818-a47b-4721-ba90-6999357fe3e2` in all three, start and end), Ubuntu
24.04 aarch64, gcc 13, CUDA 13.0.88 from NVIDIA's `ubuntu2404/sbsa` repository,
`torch 2.13.0+cu130`, on a worktree-clean checkout of the pin cloned fresh from
GitHub:

```text
$ TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=4 pip install -v --no-build-isolation -e .
FAILED: .../exllamav3/exllamav3_ext/avx2_target.o
error: ‘__builtin_cpu_supports’ was not declared in this scope;
       did you mean ‘__builtin_isupper’?   [x2, both call sites]
RuntimeError: Error compiling objects for extension
BUILD_RC=1
```

One failing translation unit, two errors, both of them the two
`__builtin_cpu_supports` calls in `avx2_target.cpp`. The build stops there
because `torch.utils.cpp_extension` runs ninja without `-k`, which is why the
next paragraph had to re-drive it to see the whole picture.

**One error in the first run was MINE and is not evidence about exllamav3.** That
run also reported `fatal error: cusparse.h: No such file or directory` three
times, because the probe's apt list omitted `libcusparse-dev-13-0`. Installing it
and rebuilding removed all three and left `avx2_target.o` as the only failure.
An instrument gap that fails toward a code verdict is exactly the trap this
project keeps hitting, so it is written down rather than quietly dropped.

**The blocker is source-level, and the build named it more completely than
reading did.** A second lease drove the same build dir with `ninja -k 0`, so
every translation unit was attempted instead of stopping at the first failure:

```text
edges the build wants   129
objects produced        122
FAILED                    7   exllamav3_ext/avx2_target.o
                              exllamav3_ext/avx512_target.o
                              exllamav3_ext/cpu/moe_handoff.o
                              exllamav3_ext/cpu/moe_mul1.o
                              exllamav3_ext/parallel/all_reduce_cpu.o
                              exllamav3_ext/parallel/all_reduce_cpu_avx2.o
                              exllamav3_ext/parallel/all_reduce_cpu_avx512.o
distinct errors           3   fatal error: immintrin.h: No such file or directory
                          3   error: ‘__builtin_cpu_supports’ was not declared
                          2   error: identifier "__builtin_ia32_pause" is undefined
```

**122 of 129 translation units compile on aarch64**, for
`-gencode=arch=compute_87,code=sm_87` under CUDA 13.0. That includes every CUDA
kernel this row mirrors — the whole of `quant/` with its 77 `comp_units`
instantiations, `dsv4_compress.cu`, `dsa_topk.cu`, `hc_mix.cu`, `routing.cu`,
`attention.cu` — and it means the GPU side of exllamav3 is not the obstacle.

**All seven failures are x86-only HOST code, and one mistake explains all of
them: `#ifdef __linux__` used where the author meant "x86".**
`grep -rn '__x86_64__\|__aarch64__\|_M_X64\|__ARM_NEON'` over
`exllamav3/exllamav3_ext/` returns ZERO architecture guards at the pin. Instead:

| site | what it does on Linux/aarch64 |
|---|---|
| `avx2_target.h:10-13` | defines `AVX2_TARGET` as `__attribute__((target("avx2")))` |
| `avx512_target.h:9-11` | defines `AVX512_TARGET` as `target("avx512f,avx512bw")` |
| `avx2_target.cpp`, `avx512_target.cpp` | call `__builtin_cpu_supports("avx2"/"avx512f")` |
| `cpu/moe_handoff.cu:151-156` | `#ifdef __linux__` -> `__builtin_ia32_pause()` |
| `parallel/all_reduce_cpu.cu:110-114` | `#ifdef __linux__` -> `__builtin_ia32_pause()` |

plus `<immintrin.h>` included unconditionally by
`cpu/moe_mul1.cpp`, `parallel/all_reduce_cpu_avx2.cpp` and
`parallel/all_reduce_cpu_avx512.cpp`. Reading the source found five of these
files; the two `.cu` spin loops only showed up when the compiler ran, which is
why the enumeration was worth the lease.

**Upstream agrees, so this is a supported-platform fact rather than a defect
report.** `.github/workflows/build.yml` declares
`oses = ["ubuntu-22.04", "windows-2022"]` and nothing else, the wheel its README
offers is `...-linux_x86_64.whl` (`README.md:88`), and the CUDA architecture list
it builds is `cudaarch = "8.0 8.6 8.9 9.0 10.0 12.0+PTX"` — which holds neither
GB10's `12.1`, nor Thor's `11.0`, nor Orin's `8.7`.

**How big the port is, measured rather than guessed.** `moe_mul1.cpp` carries a
complete scalar tier (`Isa::Scalar`, `scalar_tiles<bits>`), so guarding its AVX
regions out leaves a working fallback, and both `__builtin_ia32_pause()` sites
already have a portable shape to fall back to. The two
`all_reduce_cpu_avx*.cpp` files do not; they are the CPU all-reduce for
multi-GPU tensor parallelism, which one device never calls, but they still have
to compile. So the work is roughly: repair five `#ifdef __linux__` guards, then
find a non-x86 path for about 276 lines of intrinsics across two files. That is
a port, and it needs its own row.

**A patched exllamav3 would not be THIS oracle.** `AGENTS.md` pins an oracle so
its measurements are reproducible. A local aarch64 port changes the executing
code, so it cannot carry the pin's authority. `gateable` stays `no` until
upstream supports aarch64, or until the project deliberately registers a named,
pinned fork as its own oracle entry.

### What could NOT be measured, and why

**The GB10 itself.** `dgx:gpu0` read `unhealthy (no contact 14m47s)` at 08:02Z
after its holder's lease vanished mid-session, and `thor:gpu0` lost contact at
about 08:01Z and killed a running job of this measurement with it. Two of three
fleet devices dropped inside one hour, which is the same GB10 failure mode this
file already recorded on 2026-08-25 (`no contact 7m56s`); `dgx:gpu0` read
`no contact 1h28m25s` when this record was written, by which point `orin:gpu0`
had dropped as well and the whole fleet was unreachable.

**There is NO job waiting to supply that confirmation, and this paragraph was
wrong for about ten minutes.** A build-only job (`DO_COALESCE=0
ARCH_LIST=12.1a`, `c65ab668`) was queued on `dgx:gpu0` and this file said it
would land whenever the box returned. It did not survive: an `rc run` submission
dies with its client, so the queue entry vanished when the submitting session's
client did, and `rc ps` no longer lists it. Whoever resumes has to submit it
again rather than wait for it:

```sh
rc run -d dgx:gpu0 -- bash -c 'LABEL=dgx DO_COALESCE=0 ARCH_LIST=12.1a \
    NEED_GIB=20 bash /workspace/exllamav3-gateability/run-v3.sh'
```

That command deliberately skips the 99 GiB TP1 coalesce, because the build
verdict already rules out the run that would consume it. It also leaves
`/tmp/exl3-1901` behind on the dgx worker, which is owed a cleanup afterwards.

**That gap is narrow, because the failing code is not GPU code.**
`avx2_target.cpp` is a host C++ translation unit compiled by `g++`; it contains
no CUDA, and nothing in its failure depends on the device, the driver, or
`TORCH_CUDA_ARCH_LIST`. GB10 runs the same Ubuntu 24.04 aarch64 worker image and
the same gcc. So the BUILD half is settled for the fleet, and what a recovered
`dgx:gpu0` would add is confirmation on the row's own device rather than a new
answer.

**The RUN half was never reached**, and it is blocked behind the build rather
than by anything of its own. Two facts were established for it anyway, so
whoever picks it up does not rediscover them:

- **It would fit, in the default configuration only.** Read from the
  checkpoint's own safetensors headers: exllamav3 dequantizes DeepSeek-V4-style
  FP8/FP4 block tensors to FP16 at load (`modules/linear.py:159-178,196-205`)
  and keeps only the EXL3 trellis packed. The `text` component is then
  13.61 GiB of carried tensors plus 82.59 GiB of trellis = **96.20 GiB**,
  against the 119.63 GiB `rc describe dgx:gpu0` reports as
  `mem_total_bytes`. Adding the `mtp.*` DSpark drafter costs another
  31.34 GiB and reaches **127.54 GiB**, which does not fit.
  `Model.from_config` defaults to `component = "text"`
  (`model/model.py:153`), so the default path is the one that fits.
- **The namespaces match.** Every carried key the checkpoint ships —
  `layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}`,
  `layers.N.attn.compressor.*` (41), `layers.N.attn.indexer.*` (21, the HCA
  layers), `layers.N.ffn.gate.tid2eid` (3, matching `num_hash_layers = 3`),
  `layers.N.ffn.gate.bias` (40), `layers.N.hc_{attn,ffn}_{base,fn,scale}`,
  `hc_head_*`, `embed`, `head`, `norm` — is a key
  `architecture/deepseek_v4.py` builds, and every `no_default` config key
  `DeepseekV4Config` reads is present in the staged `config.json`. The obstacle
  is the toolchain, not the artifact.

**No `MODEL-DSV4-EXL3` gate is invalidated by this verdict**, and none is
repaired by it either. Every W1/W2 gate runs against the in-tree CPU reference
(`src/vt/cpu/cpu_exl3_dequant.cpp`) or against an independently derived
double-precision Sylvester H128, never against an exllamav3 execution. The row's
own spec says so of its single real-checkpoint anchor: those spot values "were
NOT produced by running upstream's kernel". What changes is that W3c now knows
it cannot bind a token gate to this oracle, and the SparkInfer container run is
the behavioral reference until aarch64 support exists.

```oracle-pin
id = exllamav3
role = secondary
upstream = https://github.com/turboderp-org/exllamav3
scope = the EXL3 trellis quantization format, its kernels, and the DeepSeek-V4 support the pinned HEAD carries, none of which vLLM or vLLM-Omni implements
pin = 2398c05635fbbad01a0a51dce63c85c6c8a8450e
pin_label = v1.4.3
pinned_on = 2026-08-25
gateable = no
evidence = #1901
```

**Identity, asserted rather than assumed.** The revision above was read from a
clone whose `origin` is `https://github.com/turboderp-org/exllamav3`, whose
worktree is clean (`git status --porcelain` empty) and whose `HEAD` is
`2398c05635fbbad01a0a51dce63c85c6c8a8450e`, subject "Merge branch 'pr-298' into
dev", authored 2026-08-23. `git rev-parse v1.4.3^{commit}` resolves to the same
SHA, so the pin and the tag are one commit, and `exllamav3/version.py:1` declares
`__version__ = "1.4.3"`. The repository is MIT (`LICENSE:1-3`, "Copyright (c)
2025 Turboderp"). Nothing here is a claim about upstream `master` today: this
record, like the checker that reads it, is network-free.

That clone is no longer shallow — see the ancestry measurement above — and the
identity was confirmed a second time from a different direction on 2026-08-26: a
leased worker cloned `https://github.com/turboderp-org/exllamav3` fresh, checked
out the pin, and read back `2398c05635fbbad01a0a51dce63c85c6c8a8450e` with a
clean worktree and `__version__ = "1.4.3"`. The bytes that were built are the
bytes the pin names.

## Anchors used by the port

Read at the pin above. Re-verify before relying on them.

| Piece | Anchor |
|---|---|
| `LinearEXL3`, the four owned tensors | `exllamav3/modules/quant/exl3.py:16-40` |
| TP split semantics, inverted by W1b | `exllamav3/modules/quant/exl3.py:285-329` |
| MCG codeword decode | `exllamav3/exllamav3_ext/quant/codebook.cuh:67-75` |
| codeword window unpack | `exllamav3/exllamav3_ext/quant/exl3_dq.cuh:15-31` |
| H128 + sign composition | `exllamav3/modules/quant/exl3.py:227-237` |
| `had_r_128` activation transform | `exllamav3/exllamav3_ext/quant/hadamard.cu:88` |
| `exl3_gemm` shape table | `exllamav3/exllamav3_ext/quant/exl3_kernel_map.cuh:53-62` |
| DeepSeek-V4 tensor keys | `exllamav3/architecture/deepseek_v4.py:134,167,172-174` |
| DSpark drafter | `exllamav3/architecture/deepseek_v4_mtp.py` |

The format description these anchors ground lives in
[`../specs/model-dsv4-exl3.md`](../specs/model-dsv4-exl3.md) §"The format, as
measured". This file carries identity, not methodology.
