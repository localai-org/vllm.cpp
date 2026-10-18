# Environment & assets

This file is a factual registry of known development and gate environments. It
does not authorize connecting to a host, managing its services, using its
credentials, or assuming its paths exist. The untracked
`developer-preferences.md` selects which entries are available in the current
workspace and supplies local path/lock overrides. If no profile is selected,
use only the current local host and mark unavailable hardware gates `PENDING`.

## Reaching a GPU: claim a lease, never `ssh`

The shared GPUs are managed by
[resource-controller](https://github.com/mudler/resource-controller), whose
client is `rc`. **Claim a device with `rc run` or `rc hold` before any GPU work,
and never `ssh` to a GPU box to run work directly.** A bypass makes the fleet
report the box free while somebody is on it, which is the exact failure the
lease exists to remove. The procedure is in the `leasing-a-gpu` skill, which
this repository deliberately does not copy, because a copy goes stale without
saying so.

`AGENTS.md` §`Work on a GPU happens inside a lease` holds the rule, and its
condition is the DEVICE rather than the shell you are typing in. `dgx:gpu0`,
`thor:gpu0` and `orin:gpu0` are the fleet devices, so a lease is the required
path to each of them and it replaces the file mutex as the default. The three
names are listed in both files so that membership stays checkable when the
client is not at hand, and they are a lower bound rather than an upper one: a
device that `rc devices` reports is a fleet device even when this list has not
caught up. On a GPU that is not one of them, take
`${GPU_LOCK:-$HOME/gpu.lock}` as before.

`rc devices` lists the fleet when your shell has the client AND the controller
answers, so it reports your access and not the device's membership. It fails in
at least three ways that a reader must not collapse into one: `command not
found` means this shell lacks the client, and a timeout or a refused
authentication means the controller is not answering. `thor:gpu0` read `unknown
(no contact 1m0s)` on 2026-08-17, so lost contact is a live state. On a fleet
device every one of those answers means get the client or report the controller
down. None of them means take the file mutex over `ssh`, because that is the
collision below, in which two mutexes could not see each other.

**This REPLACED the `ssh <host>` plus `flock` mechanism that the profiles later
in this file still describe.** Read a historical recipe as evidence of what ran
at the time, not as an instruction for what to run now.

**"The file mutex is still real and still required" is the stale half of this
paragraph, and the Jetson Thor profile below now contradicts it.** On a FLEET
device the lease IS the mutex — `rc` hands the whole device to one job — so a
job that also takes `${GPU_LOCK}` serialises against nobody. That profile
deletes the `flock` wrapper from its recipe for exactly that reason. The file
mutex stays real and required on a GPU that is NOT a fleet device, which is the
conditional form `AGENTS.md` states, and where both apply it runs inside the
lease rather than instead of one.

**The bypass has already voided a measurement, so this is a measured cost and
not a rule for its own sake.** `.agents/specs/minimax-music3.md` §13.10 records
a whole speed axis retained as VOID on 2026-08-17: those runs went in by `ssh`
plus `docker run` serialised by the old mutex, while a concurrent session held
the SAME box through `rc`. The two sessions took different mutexes and neither
excluded the other, which is verbatim the #777 failure, and it is the likely
cause of a 3x swing in the samples. `.agents/benchmark-record.md` records the
other half of that row taking a real `rc hold` on `thor:gpu0` and reports the
window in which the fleet showed `thor:gpu0` FREE while it was in use.

The fleet, read from `rc devices` and `rc describe` on 2026-08-17:

| Device | Labels | `/workspace` on it |
|---|---|---|
| `dgx:gpu0` | `gpu_model=GB10`, `class=train`, `k8s=true`, driver 580.173.02, `cpus=20`, 128 GB | the house NAS |
| `thor:gpu0` | `gpu_model=NVIDIA-Thor`, `class=train`, `k8s=true`, driver 595.78, `cpus=14`, 132 GB | the house NAS, the SAME folder as `dgx` |
| `orin:gpu0` | `gpu_model=AGX-Orin`, `class=train`, `k8s=true`, `cpus=12`, 32 GB, L4T R36.4.7 (JetPack 6), and NO detected GPU labels because Jetson carries no `nvidia-smi` | the house NAS, the SAME folder as `dgx` and `thor` |
| `strix:gpu0` | `gpu_model=Radeon-8060S`, `vendor=amd`, `class=train`, `k8s=true`, `cpus=32`, `gfx1151` (RDNA 3.5, AMD RYZEN AI MAX+ 395), ROCm 7.2.4 / HIP 7.2.53211, `mem_total_bytes=33270497280` | the house NAS, the SAME folder as `dgx`, `thor` and `orin` |

### `strix:gpu0` was missing from the table above until 2026-09-11

**It is the only AMD device on this fleet, and the table predates it.** The
table's own header says it was read on 2026-08-17; `strix:gpu0` is leasable,
every `BACKEND-ROCM` row leases it, and a reader who trusted the table would
conclude this fleet has no AMD hardware. Claim it with `rc run` or `rc hold`
like any other fleet device, and never by `ssh`.

**Its memory split is configurable in firmware and it was CHANGED on
2026-09-11**, so every figure recorded against this box before that date
describes a different machine. Measured inside a lease on 2026-09-11, `rc`
jobs `a8111ff8-3ce8-42f6-9034-36bdd2cacfe4` and
`c30dc437-bf12-4a23-ab8b-89b88fe767dd`:

| Probe | Value |
|---|---|
| `mem_info_vram_total` | 103,079,215,104 B = **96.00 GiB** |
| `hipMemGetInfo` total / free | 96.000 GiB / 95.848 GiB idle |
| `mem_info_gtt_total` | 16,635,248,640 B = 15.49 GiB |
| host RAM total / available | 33,270,497,280 B / 29,304,037,376 B |
| device properties | `integrated=1 managedMemory=1 pageableMemoryAccess=0 gcn=gfx1151` |

**`hipMallocManaged` on this board is bounded by HOST memory, not by the
carve.** A bounded probe that stops at its first failure reached **76 GiB with
plain `hipMalloc`** (its target, so the real ceiling is at least that and is
NOT measured above it) and **27 GiB with `hipMallocManaged`, which returned
`out of memory`**. 27 GiB against 29.3 GiB host-available is the match that
identifies which bound was hit.

This **resolves an ambiguity [#2518](https://github.com/mudler/vllm.cpp/issues/2518)
could not**. Its 58.000 GiB managed ceiling was measured when the carve was
64.00 GiB and host RAM was 62 GiB; 58 sits below both, so that number never
said which one it was. The current split separates them, and the answer is
host.

Three consequences for anyone sizing work here:

1. **Raising the carve LOWERED the managed ceiling**, 58 GiB to 27 GiB, because
   host RAM fell from 62 GiB to 31 GiB. More VRAM is not more of everything.
2. **`VT_ROCM_MANAGED_ALLOC=1` is actively harmful on this board.** The default
   is already correct: `ResolveMemoryPolicy` (`include/vt/rocm/rocm_arch.h`)
   sets `managed_alloc = pageable_memory_access` under `kUnset`, and this board
   reports that 0, so the #2511 narrowing selects plain `hipMalloc`.
3. **A large CPU arm no longer fits this box.** The `qwen4_exp` CPU arm peaked
   at 73.9 GiB `VmHWM`; the host side is now 31 GiB total. Run a CPU comparison
   on `thor` or `dgx`, or on-box against an oracle instead.

### `orin:gpu0` needs L4T CUDA 12.6, and the DGX recipe breaks it

**Do not install `cuda-toolkit-13-*` from the generic `sbsa` repo on `orin`.** That
is the correct recipe for `dgx` and `thor` and it is wrong for a Jetson. Measured
inside a lease on 2026-09-03:

| | |
|---|---|
| L4T release | `R36 (release), REVISION: 4.7` — JetPack 6 |
| driver CUDA API | `cuDriverGetVersion` = **12060**, i.e. CUDA **12.6** |
| device visible | `cuInit` rc 0, `cuDeviceGetCount` rc 0, **count 1** |
| what a job had left at `/usr/local/cuda` | `cuda-13.3` |

The board is healthy and the driver sees the GPU. A CUDA **runtime** of version N
requires a driver API of at least N, so a 13.3 toolkit against a 12.6 driver makes
`cudaGetDeviceCount` return **35** (`CUDA_ERROR_INSUFFICIENT_DRIVER`). That 35 is
correct behaviour reporting a packaging mistake, not a broken board, and it is why
`orin` has read as "cannot run CUDA".

The OS image is not at fault and needs no change. `kairos-init` already pins CUDA
**12.6** for `nvidia-jetson-agx-orin` from the L4T repo
(`repo.download.nvidia.com/jetson/{common,t234}` at `r36.4`) and moves it to
`/opt/cuda-12.6`; no Dockerfile in the `dockerfiles` repo installs a toolkit at
all. The 13.x came from a JOB, and the worker container is long-lived, so one
job's global install is still there for the next one.

**What to do in a job on `orin`:** use the CUDA already on the board, or install
`cuda-*-12-6` from the jetson repo. Never `developer.download.nvidia.com/.../sbsa`
and never `cuda-toolkit-13-*`. **Compilation is unaffected** — `nvcc` needs no
matching driver, so a 13.3 toolchain builds this tree for `sm_87` cleanly; only
running does.

**Select on `class` or `gpu_model`, never on `vram`.** `rc describe` reports
`vram=[N/A]M` and `vram_free=[N/A]M` on this fleet. That is a probe reporting
"unknown", not a value, so a selector such as `vram>=40G` matches nothing and
the job is rejected with `no_matching_device`. A device that carries no label
never matches, INCLUDING for `!=`, and `orin:gpu0` carries no detected GPU
labels at all. `class=train` and `gpu_model=GB10` do match. `rc run` has no
`--image` flag. Its flags are `--as`, `--cwd`, `-d`, `--explain`,
`--idle-timeout`, `--max-runtime`, `--no-wait`, `--priority`, `--select` and
`--timeout`.

### What the `dgx:gpu0` leased worker can and cannot do, measured 2026-08-17

Probed with one `rc run -d dgx:gpu0 --max-runtime 2m` job
(`ff28ada1-0cd3-4867-bf9b-f67050d0608b`). Verify this again before you plan work
around it, because the worker image can change under you. **It did change.** The
`thor:gpu0` worker measured later the same day carries `python3` and `gcc`, which
this list calls absent, so read this section as one box on one day. The `thor`
reading is in "A relocated CUDA runtime starts on `thor:gpu0`" further down.
**The `dgx:gpu0` worker then changed as well**, so read the whole section as
history and take the toolchain from
[#1213](https://github.com/mudler/vllm.cpp/issues/1213).

- The command runs as user `rc` in a **k3s pod**, hostname `rc-worker-<id>`.
  `/.dockerenv` is absent and 8 `KUBERNETES_*` variables are set, so it is a pod
  rather than a docker container. The toolchain question is therefore a
  worker-image question, not a per-job one.
- Present: `bash`, `sh`, `ls`, **`nvidia-smi`** (which reports the GB10 by UUID),
  `flock`, and **`/workspace`**.
- **Absent on 2026-08-17: `gcc`, `cc`, `clang`, `nvcc`, `ninja`, `cmake`,
  `make`, `python3`, `python`, `pip`, `docker`, `sudo`, `git`, `ssh`, `curl`,
  `/usr/include/stdio.h`, and any `/usr/local/cuda*` toolkit.** That probe read
  the worker as unable to compile, to start Python, or to install anything.
  **That reading is SUPERSEDED.** `rc describe dgx:gpu0` now states that the job
  runs as root in an Ubuntu 24.04 container carrying `git`, `curl`, `wget`,
  `ssh`, `gcc`, `g++`, `make`, `cmake`, `ninja`, `pkg-config`, `python3`, `pip`
  and `venv`, and it instructs the reader to install anything missing. The one
  limit the sheet names is the CUDA toolkit, which a job apt-installs per run
  ([#1213](https://github.com/mudler/vllm.cpp/issues/1213)).
  **Do not carry a one-box reading to another device either.** On `thor:gpu0`
  the same day the worker ran as `uid=0(root)` with `/usr/bin/gcc`,
  `/usr/bin/python3` and a working `apt-get`
  ([#1146](https://github.com/mudler/vllm.cpp/issues/1146)).
- **A leased worker's `github.com` egress is not guaranteed, and its absence
  reads as an authentication failure rather than as a network one.** On
  2026-09-02 a `thor:gpu0` job that had fetched llama.cpp from `ggml-org` earlier
  the same day failed at `git fetch` with `could not read Username for
  'https://github.com'` followed by `fatal: expected flush after ref listing`
  (rc job `019596fd-5e88-4d0d-aa93-2749ab618524`). Nothing about that message
  says "no route"; a reader chasing credentials would look in the wrong place.
  **A job that needs pinned upstream source should stage it rather than fetch
  it**, and staging costs no rigour if the tree is verified rather than named:
  `git archive <PIN>` on the dev box, then `git init && git add -A --force &&
  git write-tree` on the worker, asserted equal to `<PIN>^{tree}`. That rehashes
  every blob from the bytes that get compiled, which is a stronger statement than
  `git rev-parse HEAD` plus an empty porcelain
  ([#2534](https://github.com/mudler/vllm.cpp/issues/2534),
  `docs/bench-evidence/qwen38-27b-q4km-oracle-path-pin-20260902.md`).
- **The host filesystem is not visible.** `/home/mudler` does not exist inside
  the worker.
- `/workspace` is the house NAS, measured as `//192.168.68.102/Data 7.3T total,
  4.0T available, 46% used`, writable from the job, mounted on the dgx host at
  `/usr/local/nas_share/rc` (SMB, NodePort 31516, subfolder `rc/`). It is the
  SAME folder from `dgx`, from `thor` AND from `orin`, and it is the one surface
  every end can see.

**The consequence, and it is now narrower than a blocker.** The pinned oracle
venv lives at `~/venvs/vllm-oracle-pin-555967922` on the dgx HOST, and a leased
worker cannot see it. The dgx host has carried no toolchain since the 2026-08-14
reimage, so host-side oracle work needs `sudo -n docker run` against
`vllmcpp-build:gb10` or `nvidia/cuda:13.0.1-devel-ubuntu24.04`, reached over
`ssh`, which is the bypass. **The sentence this paragraph used to carry, "no
vLLM leg of any row can currently run on `dgx.casa` by a lease-compliant path",
is FALSIFIED, for the BUILD step and for a MODEL RUN alike.** On 2026-08-18
two `rc run` jobs built the pin from source inside a lease, installed the wheel,
imported it, and reported `cuda True NVIDIA GB10`
([#1185](https://github.com/mudler/vllm.cpp/issues/1185), and "The pinned oracle
builds inside a lease on `dgx:gpu0`" further down). On 2026-08-19 the same pin
then SERVED: `vllm serve` on a 52 GiB bf16 checkpoint, from a lease, no `ssh` and
no container image, three clean benchmark legs (`.agents/benchmark-record.md`,
`BENCH-QWEN38-27B-BF16` c1, and "A model DOES run inside a lease" below). So
oracle-side MEASUREMENT from a lease is no longer blocked; what is still
unreachable is the image-based path SGLang needs
([#1265](https://github.com/mudler/vllm.cpp/issues/1265)). Read the old reason
carefully before you quote it, because it was never the worker's missing
toolchain. "The lease carries bytes, and the exec bit is a mount option" below
measures staged content starting under the dynamic loader and after a copy to
`/tmp`. That is why recent GPU work reached for `ssh`, and the bypass is a
symptom of this gap rather than a discipline problem. Do not design the
migration here. `ENV-LEASE-RUNTIME-STAGING` owns the runtime staging, and
[`lease-runtime-staging.md`](specs/lease-runtime-staging.md) holds its working
recipe for `torch` and `triton`. `ENV-ORACLE-WHEEL-IN-LEASE` owns the oracle
build, in [`oracle-wheel-in-lease.md`](specs/oracle-wheel-in-lease.md).

**This confirms and extends a finding that already landed, rather than making a
new one.** `.agents/specs/minimax-music3.md` §13.10 probed `thor`'s worker on
2026-08-17 and found the same absence (`no gcc / g++ / cmake / ninja / nvcc /
make`), reported that the `$HOME` build tree is not mounted inside the worker,
and named what a valid re-measurement needs: either a worker image carrying the
CUDA devel toolchain, or the build placed on `/workspace` by something that
already has one. This row measured `dgx`'s worker and adds the part that turns
an open gap into a blocker for the parity gates, which is that the ORACLE VENV
is also unreachable from a lease.

### Clock pinning does NOT work inside an `rc` lease, measured 2026-08-19

[#1354](https://github.com/mudler/vllm.cpp/issues/1354) owns this.

`nvidia-smi -lgc` is refused inside the leased worker, in a job running as
**root**:

```text
$ nvidia-smi -lgc 2190
The current user does not have permission to change clocks for GPU 0000000F:01:00.0.
LGC_RC=4
```

Reproduced in three separate `rc run` jobs on `dgx:gpu0` on 2026-08-19
(`.agents/benchmark-record.md`, `BENCH-QWEN38-27B-BF16` c1/c8). The container is
root but lacks the capability the driver requires.

**This outlives one campaign, and it is a records defect as much as a capability
one.** [`benchmarking.md`](benchmarking.md) instructs "Pin the clocks before
measuring, under the lock", and **every clock-pinned figure in this
repository's records was taken over the retired host + `ssh` + `flock` path**.
The migration to `rc` leases silently removed clock pinning and no record said
so. Same class as [#1265](https://github.com/mudler/vllm.cpp/issues/1265): a
capability the records assume, which the current access path does not provide.

**Consequence for anyone measuring from a lease.** The SM clock can only be
SAMPLED, never pinned. `tools/bench/gpu_clock_state.py` still works and is the
only attribution such a number carries. Plan for its within-run rule to fail:
the 2026-08-19 series recorded within-run spreads of 12.92% to 26.36% across
nine windows on a thermally throttling GB10, against the 5% ceiling, and
`gpu_clock_state compare` therefore returned `PAIRING_VERDICT=DISCARD` on every
c1 pairing even though the cross-arm rule passed perfectly. Two arms measured in
a lease may not be dividable, so budget for absolutes and say so in advance
rather than discovering it after the GPU time is spent.

**The missing capability is `CAP_SYS_ADMIN`, and it is missing from the BOUNDING
set. Measured 2026-08-21 inside leases on `thor:gpu0` and `orin:gpu0`.**
`/proc/self/status` in the worker reads `CapEff = CapPrm = CapBnd =
0x00000000a80425fb` on both boxes, byte-identical -- the default 14-capability
OCI set, which holds neither `CAP_SYS_ADMIN` nor `CAP_PERFMON` nor
`CAP_SYS_NICE`. Those three are the whole Linux privilege surface of the NVIDIA
open kernel module: at tag `580.173.02`, `grep -n 'capable(\|CAP_'` over
`kernel-open/common/inc/nv-linux.h` and `kernel-open/nvidia/os-interface.c`
returns `NV_IS_SUSER() == capable(CAP_SYS_ADMIN)` at `nv-linux.h:537`,
`capable(CAP_PERFMON)` at `os-interface.c:390`, and `capable(CAP_SYS_NICE)` at
`os-interface.c:397`, and nothing else. **`CapBnd` is why no job can route around
it**: a capability absent from the bounding set cannot be regained by `setcap`,
by a setuid binary, or by re-execing, so this is container configuration and not
job authorship. `thor:gpu0` reproduced the refusal itself on driver `595.78`
(`LGC_RC=4`), so the refusal is measured on two boxes and two driver versions.
`/proc/driver/nvidia/params` reads `RmProfilingAdminOnly: 1` in the lease, which
governs PROFILING COUNTERS and is a different gate -- do not reach for an `NVreg`
knob expecting `-lgc` to start working. The ask, the acceptance test that
falsifies it, and the cost are in
[`lease-gpu-capability.md`](specs/lease-gpu-capability.md).

**Clock pinning DOES work from the leased HOST, and the pod path is still
refused. Measured 2026-08-21.** Under an `rc hold` lease on `dgx:gpu0`, over
`ssh` to the host, `sudo -n nvidia-smi -pm 1` then `-lgc 2100` held **0.29%**
SM-clock spread over a ten-minute decode load (120 samples, min 2080, max 2086),
against the 12.92%-26.36% the unpinned windows above recorded. **That path needs
an authorization most rows do not have**: `.agents/developer-preferences.md`
scopes `rc hold` plus `ssh` to the `BENCH-QWEN38-27B-SOTA` campaign and leaves
the standing rule unchanged for every other row. So a ratio is derivable on that
box by that path, and the pod path -- the only path for every other row -- still
returns `LGC_RC=4`. Two caveats travel with the host path: persistence mode had
to be enabled first, and it can be lost when the last GPU client detaches, so
**verify the clock DURING the run and never only before it**; and the recorded
`-lgc 2100` recipe is about 69% of this device's 3003 MHz maximum SM clock, which
is right for a RATIO and wrong for an ABSOLUTE.

**`nvidia-smi -pm 1` DOES work inside the pod, and it is a DIFFERENT knob from
`-lgc`. Measured 2026-08-22 on `dgx:gpu0`.** The `SPEC-DFLASH2` speed gate (job
`ec9cf6cd-0aaf-4323-806d-6a12da2bd08f`) found `persistence_mode: Disabled` when
it opened, because the box had rebooted — `boot_id` moved from `db4ca4f3-...` to
`302145bc-4c57-4f78-803c-f9d644a24b9d` — and a reboot resets the driver setting.
The job ran `nvidia-smi -pm 1` INSIDE the lease, as a normal leased job with no
`ssh` and no host access, and it SUCCEEDED: "Enabled persistence mode via daemon
for GPU 0000000F:01:00.0." Both arms then sampled `persistence_mode: Enabled`.
`-lgc` in the same job still returns 4, so this does not weaken #1354 and does
not make clock pinning reachable; the two are separate driver operations and
only one of them is refused.

**Why this matters enough to write down: without it the lease produces
NOTHING.** `gpu_clock_state.clock_reasons` appends a refusal for any record
whose `persistence_mode` is not `Enabled`, so both arms of a two-hour paired
measurement would have been discarded on a setting one command fixes. **Read
persistence mode at the START of a leased measurement and set it if it is off**,
rather than discovering it in the refusal. Evidence:
`/mnt/nas_share/rc/dflash2-1673/out-n1673b/m-pm.log` and the `DEVICE STATE`
paragraph of `/mnt/nas_share/rc/dflash2-1673/RUN-PROVENANCE.txt`.

**A settle-and-hold procedure CAN reach the clock gate on this box, and what
controls it is the window's REQUEST COUNT. Measured 2026-08-23 on `dgx:gpu0`.**
Without any pin, a window holding **one** request of 1024 input and 950 output
tokens cleared the 5% spread rule and the throttle rule together in 2 of 3
attempts. A window holding **six** requests of 1024 input and 128 output tokens
cleared them in 0 of 3. Each request carries one SM-clock excursion about 3.6 s
after its own start, and the excursion carries a non-benign throttle bit with
probability 0.476, so a window's chance of being clean falls as `(1-p)^requests`.
Heat is not the discriminator: the labelled-sample rate is flat at 0.9-1.0 per
minute over the last three quarters of the job, whose busy rows span 69 C to
85 C.
**No pairing was established**, because a pairing needs both arms clean and
because that run never set persistence mode. Two traps travel with it. The job's
own thermal summary printed every 63rd row and therefore showed 0 of the 34
throttle-labelled rows, a 77 C maximum against the true 85 C and a 43.9 W
maximum against the true 81.1 W, so read `thermal.csv` and never the summary
printed from it. And a window shaped to satisfy the gate measures a different
workload than the one a ratio owes. Evidence and derivation:
[`specs/clock-gate-route.md`](specs/clock-gate-route.md) §The 2026-08-23 settle
run, raw files at `/mnt/nas_share/rc/clk1354/out/settle-20260823T004328Z/`.

**A model DOES run inside a lease.** The same series ran the pinned oracle
`0.1.dev1+g555967922` as a server on a 52 GiB bf16 checkpoint from a lease, no
`ssh` and no container image, and it served three clean benchmark legs. That
retires the "nobody has run a model that way" reading of
[#1185](https://github.com/mudler/vllm.cpp/issues/1185) recorded above. What is
still unreachable is the image-based path SGLang used
([#1265](https://github.com/mudler/vllm.cpp/issues/1265)). A wheel route around
it is specified by row `SGLANG-ORACLE-LEASE-WHEEL` in
[`sglang-wheel-in-lease.md`](specs/sglang-wheel-in-lease.md), which needs no
source build and no image. **That route RAN on 2026-08-23**: two `rc` jobs on
`dgx:gpu0` installed the pinned wheels, asserted the installed tree against a
3338-file manifest at `IDENTITY_RC=0`, served Qwen3.8-27B bf16 to readiness in
454 s and completed a c1 leg with 6 of 6 requests, 0 errors and exactly 768
output tokens. The SGLang oracle is `gateable = yes` again, on the wheel route
and not on the image. The image path stays forbidden.

**And one instrument rule, paid for in the same series.** A guard whose
threshold sits inside the guarded configuration's own operating point
manufactures the finding it was meant to detect. A 12,000 MB `MemAvailable`
watchdog killed a healthy `vllm serve` 18 s after it reached `/health`, which
read as "the denominator collapses in a lease" — and the arithmetic refuted it:
`0.85 x 122,502 MB` reserved by design predicts 12,223 MB free against an
observed floor of 11,917 MB, a 306 MB match, and the same server later ran three
clean legs at a ~7,500 MB steady state. **A tripped guard is evidence about the
GUARD until its threshold is shown to sit outside the guarded thing's operating
point.** Predict the number from the configuration before believing the verdict,
and keep instrument thresholds strictly apart from engine knobs: moving the
floor changes nothing about what is measured, and moving
`gpu_memory_utilization` changes what the number means.

### The lease carries bytes, and the exec bit is a mount option, measured 2026-08-17

Probed with two `rc run -d dgx:gpu0 --max-runtime 3m` jobs,
`1cb56f84-62bf-4c90-b138-9bd4c3b0617a` and
`c692d5a0-ec3d-4498-86e4-e86a2864e91a`. Verify this again before you plan work
around it, because the worker image and the mount options can change under you.

`/workspace` in the worker and `/mnt/nas_share/rc/` on a local host are the same
folder. A file written from the worker appeared locally under that path, and a
file staged locally was read by the worker. The worker's `df` reported
`//192.168.68.102/Data`, 7.3T total, 4.0T available, 46% used.

**Direct execution off `/workspace` is refused, and the mount causes it rather
than a `noexec` flag.** The worker mounts the share with `file_mode=0664`,
`dir_mode=0775`, `nounix`, `forceuid` and `forcegid`, and the option list holds
no `noexec`. The same bytes read `-rwxr-xr-x` on the local host, which mounts
the share with `file_mode=0755`, and `-rw-rw-r--` in the worker, so the exec bit
is a presentation of each mount and not stored state. In the worker, `chmod +x`
failed with `Operation not permitted`, and a staged shell script and a copied
ELF binary each failed to start with `Permission denied` and exit code 126.

**Three routes ran staged content anyway, and each measured green.** Record the
distinction, because a missing exec bit reads like a wall and is not one.

| Route | Measured |
|---|---|
| `sh /workspace/staged.sh` | printed the script's output, exit 0 |
| `/lib/ld-linux-aarch64.so.1 /workspace/echo_copy` | ran the ELF, exit 0 |
| `cp` to `/tmp`, then `chmod +x`, then run | ran the script and the ELF, exit 0 |

`/tmp`, `/var/tmp` and `$HOME`, which is `/home/rc`, are writable, take a real
exec bit, and sit on a 3.6T overlay with 2.5T available. `/dev/shm` is 64M. The
job's working directory `/` is not writable.

**So the lease carries bytes, and bytes are enough to run.** A runtime staged on
`/workspace` can start under the dynamic loader, or after a copy to `/tmp`.
**This paragraph used to add that the `dgx:gpu0` worker cannot produce or fetch
that runtime, because it had no `curl`, `wget`, `git`, `gcc`, `nvcc`, `cmake` or
`python3`. That clause is SUPERSEDED.** The worker runs as root and carries every
one of those names except the CUDA toolkit, and a job apt-installs `nvcc` from
`developer.download.nvidia.com` per run
([#1213](https://github.com/mudler/vllm.cpp/issues/1213)). Present and useful
for staging: `cp`, `cat`, `tar`, `chmod`, `perl`, `flock` and `nvidia-smi`. **The
`thor:gpu0` worker produces one as well**, because it is root and carries
`apt-get` and `gcc`. That is the section below.

**Checkpoints: stage NAS -> local ONCE through `scripts/rc-stage-checkpoint.sh`,
never read a gate model off `/workspace`** (developer direction 2026-08-23,
[#1807](https://github.com/mudler/vllm.cpp/issues/1807)). A 22 GB or 67 GB
safetensors checkpoint read over CIFS pays the bandwidth on every shard every
run, and the HF-cache layout cannot live on the share at all (no symlink). The
gate checkpoints are staged under `/mnt/nas_share/rc/ckpt/<name>/` in HF
`--local-dir` layout (plain files), which a lease sees as
`/workspace/ckpt/<name>/`, each beside a `SHA256SUMS` manifest written once with
`scripts/rc-stage-checkpoint.sh --make-manifest /mnt/nas_share/rc/ckpt/<name>`.
A job then runs

```sh
scripts/rc-stage-checkpoint.sh /workspace/ckpt/<name> /tmp/ckpt/<name>
export VT_QWEN36_SNAPSHOT=/tmp/ckpt/<name>   # or the gate's own override
```

and the copy is idempotent and verified: a second run with a complete local
copy reads only the manifest and exits 0 (`ALREADY STAGED ... nothing read`),
a killed copy resumes from the files that verified, a local file whose bytes
differ is replaced, and a directory with no manifest is refused rather than
trusted. Staged on the NAS this way: `qwen36-35b-a3b-nvfp4`
(`nvidia/Qwen3.6-35B-A3B-NVFP4` @ `491c2f1ea524c639598bf8fa787a93fed5a6fbce`)
and `qwen36-35b-a3b-bf16` (`Qwen/Qwen3.6-35B-A3B`), both for the 35B gates.

**This narrows [#1129](https://github.com/mudler/vllm.cpp/issues/1129) and does
not close it.** The HOST venv at `~/venvs/vllm-oracle-pin-555967922` stays
unreachable from a lease, and only a host-side actor reached over `ssh` can
place a copy of it on the NAS. That route is no longer the only one, because a
lease BUILT the pin on 2026-08-18 rather than copying it
([#1185](https://github.com/mudler/vllm.cpp/issues/1185)). **Whether a relocated CUDA runtime then starts is no longer
UNMEASURED. It starts, on `thor:gpu0`.** The section below has the reading. A
CUDA virtual environment still holds absolute paths in its shebangs and its
`RECORD` files, so a `pip install --target` tree is the shape that was measured
and a copied venv is not.

**Three fleet-side changes would each remove the staging problem, and none of
them is ours to make.** Whoever owns the fleet picks one. **A fourth route was
then measured, and it needs nobody's permission:** the `thor:gpu0` worker runs
as root with a working `apt-get`, so a job provisions its own container.

1. The worker image gains a toolchain and a Python interpreter.
2. `rc run` gains an `--image` flag, so a job selects an image that has them.
3. `/workspace` is mounted so that a file there can carry an exec bit. This one
   removes the copy step only, because the two routes above already execute.

### A relocated CUDA runtime starts on `thor:gpu0`, measured 2026-08-17

Probed with six `rc run` jobs on `thor:gpu0`: `6f4bdb03`, `9c0ebeac`, `8beba132`,
`f60d945f`, `63c60a90` and `fd5654c0`. A `torch`, `triton` and `numpy` tree
staged on `/workspace` imports, initializes CUDA, runs a bf16 matmul, and
compiles and executes a Triton kernel. The job IDs in full, the staged-script
sha256 values, the four walls and the working recipe are in
[`lease-runtime-staging.md`](specs/lease-runtime-staging.md)
([#1146](https://github.com/mudler/vllm.cpp/issues/1146)).

```
torch.__version__= 2.13.0+cu130      cuda available = True
device 0         = NVIDIA Thor       capability     = (11, 0)
triton.__version__ = 3.7.1           TRITON_JIT_OK  = 4096.0 PASS
```

The recipe, once per worker container:

```sh
apt-get update -qq && apt-get install -y -qq python3-dev
mkdir -p /tmp/tp && cp -a /workspace/oracle-probe/site/triton /tmp/tp/
chmod -R +x /tmp/tp/triton/backends/nvidia/bin/

export PYTHONPATH=/tmp/tp:/workspace/oracle-probe/site
export CPATH=/workspace/oracle-probe/pyhdr/python3.12:${CPATH:-}
```

**Read the scope before you quote it.** This is `thor:gpu0` at capability (11,0)
and nothing else. The GB10 is `sm_121a`, and this tree was never staged there,
so nothing here licenses a claim about the Spark. Only `torch`, `triton` and
`numpy` are in this tree, so the pinned vLLM oracle is not shown to run by it.
The clause this paragraph used to carry, that the oracle "needs `nvcc`, which
the worker lacks", is FALSIFIED: a `dgx:gpu0` lease built the pin against a
staged CUDA toolkit on 2026-08-18, in the section after the next one. The torch
wheel is `+cu130` while the staged `ptxas` reports `release 12.8, V12.8.93`, and
that skew is recorded as observed rather than adjudicated.

**A prebuilt wheel does not remove the `nvcc` requirement, and that is measured.**
An aarch64 vLLM wheel exists in general: `pip download --no-deps vllm` on the
worker fetched `vllm-0.27.1-cp38-abi3-manylinux_2_28_aarch64.whl`, 307,180,998
bytes. Our pin is not reachable that way, because
`https://wheels.vllm.ai/nightly/vllm/` lists wheels for exactly ONE commit and is
a moving pointer rather than an archive, and because the pin is a development
version that is not on PyPI. Four 404s under a per-commit URL scheme were also
seen, and they prove nothing, because that scheme was never confirmed against a
known-good case. So reproducing the pinned oracle needs a source build or a
deliberate pin advance. Nobody established that vLLM never retains per-commit
wheels. That source build was then run inside a lease and produced our own
wheel, so the route this paragraph names is open.

### The pinned oracle builds inside a lease on `dgx:gpu0`, measured 2026-08-18

Two `rc run` jobs on `dgx:gpu0`. The pinned vLLM oracle builds from source,
installs, imports, and sees the GPU inside a lease. No raw `ssh` was used, so
the fleet reported the box as held for the whole window. The job details, the
staged-script hashes, the four staging walls and the non-claims are in
[`oracle-wheel-in-lease.md`](specs/oracle-wheel-in-lease.md)
([#1185](https://github.com/mudler/vllm.cpp/issues/1185)).

```
HEAD             = 5559679229bc961848b121ccdeaa8fa5d79bec98   PIN CONFIRMED
nvcc             = release 13.3, V13.3.73    NVCC_RC=0
wheel            = vllm-0.1.dev1+g555967922.cu133-cp312-cp312-linux_aarch64.whl
vllm.__version__ = 0.1.dev1+g555967922       IDENTITY_RC=0
cuda True NVIDIA GB10                        CUDA_RC=0
```

The `nvcc` came from the CUDA toolkit that row `MODEL-NEMOTRON-H-ABI-A3-E2E`
staged on the NAS. The build script asserts `HEAD` against the pin before it
compiles anything, and it aborts when the two differ.

**Read the scope before you quote it, because three things are UNMEASURED.**
Running a model is untested: only the build, the install, the import and
`torch.cuda.is_available()` were measured, and the recorded failure mode of the
step after `torch.compile` on this host is a reboot of the box
(`.agents/specs/mtp-k-gt-1.md`). The wheel reports `0.1.dev1+g555967922` while
`.agents/upstream-sync.md` records
`vllm_runtime_version = 0.23.1rc1.dev1511+g555967922`, an OPEN discrepancy whose
cause is the shallow fetch that stops `setuptools_scm` counting the commits
since the last tag. The virtual environment is NOT staged, because that job was
killed at its 90-minute ceiling and its partial tree was removed, so only the
WHEEL is durable.

**Two staging traps are worth carrying forward.** A `cp -a` from the NAS
preserves `file_mode=0664`, so `nvcc` exits 126, and CIFS `nounix` stores no
symlink, so a copied CUDA toolkit loses `include`, `lib64` and 32 library links
and CMake then reports
`Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) (found version "13.3")`,
which names the version and denies the toolkit in one line. **The `rc` worker
container is REUSED between jobs**, so a repair inside a staging branch is
skipped on the next run and reports `nvcc already in place`. Write an
environment repair unconditionally, and assert its postcondition.

### A staged CUDA toolkit links only if its SONAMEs were rebuilt, measured 2026-08-28

The section above records that CIFS stores no symlink and that a copied toolkit
therefore loses its library links, with CMake reporting
`Could NOT find CUDA` as the symptom. **There is a second, quieter symptom of the
same cause, and it costs a whole build rather than eleven seconds.**

A staging branch that rebuilds `libcudart.so` and `libcublasLt.so` -- the
DEVELOPMENT links -- satisfies CMake completely. `Found CUDAToolkit` succeeds,
every CUDA translation unit compiles, and the job dies ~21 minutes later linking
the first consumer:

```
/usr/bin/ld: libvllm.so.0.0.3: undefined reference to `cudaStreamSynchronize@libcudart.so.13'
/usr/bin/ld: libvllm.so.0.0.3: undefined reference to `cublasLtMatmul@libcublasLt.so.13'
... 38 in total, every one @libcudart.so.13 or @libcublasLt.so.13
```

`libcudart.so.13` is the **SONAME**, a THIRD name distinct from both
`libcudart.so` and `libcudart.so.13.3.29`, and it is the name the linker resolves
versioned undefined symbols against. In a real install it is a symlink, so CIFS
does not carry it and a staging branch must recreate it explicitly.

**The trap inside the trap is the parameter expansion.** `${f#*.so.}` strips the
SHORTEST prefix, so for `libcudart.so.13.3.29` it yields `13.3.29` and not `13`:

```sh
b=${f%%.so.*}; ln -sf "$f" "$b.so.${f#*.so.}"   # links the file to ITSELF
```

That line looks like it makes the version link and makes nothing. Take the major
with `v=${f#*.so.}; ${v%%.*}`, or better, let `ldconfig -n <libdir>` read each
object's own `DT_SONAME` so the name cannot disagree with what the linker asks
for. `ldconfig -n` does NOT create the `.so` development link, so both are needed.

**Assert the postcondition, and assert the one the failure depends on.** The
harness that hit this checked `[ -f .../libcublasLt.so ]`, which is precisely the
link its own reconstruction created correctly -- so the precondition passed on a
toolkit that could not be linked against. Check that `<stem>.so` resolves AND
that `<stem>.so.<MAJOR>` exists, for `libcudart` and `libcublasLt` both. That
check costs a second and it discriminates: run against the CIFS source tree it
FAILS, which is the correct answer.

**This is latent on any box that already has a toolkit.** The staging branch is a
fallback. `dgx:gpu0` carried `/usr/local/cuda` 13.0.88 for every earlier lease and
the fallback was never taken; the box went `unhealthy … worker_lost` for 3h20m on
2026-08-28 and came back without it, which exercised the branch for the first
time. `rc` job `1ad519b1-4e75-41d7-9386-9932076390f1`, exit 34,
[#2220](https://github.com/mudler/vllm.cpp/issues/2220).

### `ccache` is mandatory here and the usage sheet puts it where it cannot work, measured 2026-09-01

**Do not put `CCACHE_DIR` on `/workspace`.** `rc describe <device>`'s host-wide
usage sheet requires `ccache` for every supported C, C++, CUDA and HIP build on
this fleet, and instructs `export CCACHE_DIR=/workspace/ccache`. That instruction
is wrong. Follow this section instead, and read the sheet's requirement to *use*
ccache as still binding.

The mechanism is the SONAME defect above ([#2220](https://github.com/mudler/vllm.cpp/issues/2220))
arriving at a second consumer. `/workspace` is CIFS mounted `nounix`, so
`symlink(2)` on it returns `EOPNOTSUPP`. `ccache` 4.9.1 takes every cache **and
stats** lock by creating a symlink, so on that mount nothing is stored and no
counter is written:

```text
Could not acquire /workspace/.../ccdir/4/c/stats.lock: Operation not supported
Failed to acquire lock for /workspace/.../ccdir/4/c/stats
```

**`ccache -s` then reports zero hits, zero misses AND zero stores, and that
reading points the wrong way.** A cache that was consulted and empty records
MISSES. Zero of everything looks like proof that the launcher never reached the
compile line; it in fact means every write was refused. `rc` job
`93a60151-7d4d-4718-842c-ef724208be0e` passed all three
`-DCMAKE_*_COMPILER_LAUNCHER=ccache` flags, had `ccache` on `PATH`, and paid its
**1404 s (23.4 min)** build in full. Its `ccache-before.txt` and
`ccache-after.txt` are byte-identical. `/workspace/ccache` still holds the
residue: a complete 16x16 bucket tree, mtimes from inside that build's window,
**one** file and zero bytes.

**It is worse than doing nothing.** Each refused acquisition costs a retry
timeout of roughly 0.4 to 0.6 s, and `ccache -s` walks all 256 stats buckets.
Against a `/workspace` cache, `ccache -s -v` produced no output at all inside the
probe window after 278 consecutive failures and the job had to be killed. A cache
there adds time to every build it cannot accelerate.

**The recipe. `CCACHE_DIR` local, the persistence on the NAS.**

```sh
command -v ccache >/dev/null || { apt-get update -qq; apt-get install -y -qq ccache; }
export CCACHE_DIR=/root/ccache                              # local: it needs symlink(2)
export CCACHE_REMOTE_STORAGE=file:/workspace/ccache-remote  # survives the container
export CCACHE_MAXSIZE=20G
mkdir -p "$CCACHE_DIR" "${CCACHE_REMOTE_STORAGE#file:}"
cmake -S "$SRC" -B "$BLD" -G Ninja ... \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
```

The remote store MAY live on `/workspace`, and the difference is not a guess:
ccache's `file` backend stores through `open` plus `rename`, which this mount
serves, and takes no lock at all. Measured on `strix:gpu0`, `rc` job
`e4793984-10e8-4b2e-b6a7-7f931d6d40fe`, over eight translation units:

| Arm | `CCACHE_DIR` | Remote store | Result |
|---|---|---|---|
| local | `/tmp` | none | healthy: 1 hit, 1 miss over 2 calls |
| cifs | `/workspace` | none | 278 lock failures; `ccache -s -v` never returned |
| remote-seed | `/tmp`, empty | `file:/workspace/...` | 18 s, 0/8 hits, 16 remote writes |
| remote-cold | `/tmp`, **fresh and empty** | the same store | **5 s, 8/8 remote hits, 0 misses** |

`remote-cold` is the arm that carries it: its local cache was created empty, so
every hit came off the NAS. A tarball of the local cache on `/workspace` also
works (seed 4 s, save 1 s, restore 0 s, warm 2 s) and is the fallback if remote
storage is ever unavailable; it is not the default because it needs an explicit
save step that a job killed by `--max-runtime` never reaches.

**The three causes this is NOT, each excluded by the same probe** (`rc` job
`9b287a1f-d94d-476d-9732-0f649509cdd8`). The CMake launcher DOES reach the
compile line: `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` put `ccache` on 9 lines of a
verbose Ninja build and produced `Cacheable calls: 9 / 9 (100.0%)`. `ccache` IS
on `PATH` and resolves to `/usr/bin/ccache`. No wrapper or generator bypasses it.
Only the filesystem is at fault.

**Assert that the cache took part, rather than logging its counters.** A line
that prints hits and continues is what let this run for a month.
`scripts/ltx25-render-confirm.sh` now probes `symlink(2)` on `CCACHE_DIR` before
the build and exits 37 if it fails, and refuses after a build that compiled if
`Cacheable calls` is zero. The assertion is on cacheable CALLS and not on hits,
because an honest first build has every right to zero hits.
`scripts/check-lease-ccache.py` holds the rule for every script under `scripts/`.

### Two packages a DFlash2 oracle lease needs, and the lease variable that exists, measured 2026-08-22

Measured on `dgx:gpu0` across leases `11cee02a`, `52ac5673` and `a03f34e4`
([#1660](https://github.com/mudler/vllm.cpp/issues/1660)). This is the "install
what you lack" instruction above, made specific for the one job that has paid
for the gap.

**`cuda-libraries-dev-13-0`, and the failure arrives 12 minutes late.**
DFlash2's `compute_candidates` -> `_topk` -> `flashinfer.topk` JIT-compiles
`topk.cu`, which includes `<curand.h>`. The `cuda-toolkit-13-0` metapackage does
NOT install that header. Leg B died on it INSIDE `profile_run`, after a
12-minute model load, and it presented as a model failure rather than as a
missing header; leg C installed `cuda-libraries-dev-13-0` and got past it. A job
that compiles CUDA therefore installs both, not just the metapackage.

**`python3-dev`, and it lies about what failed.** Without it Triton's driver JIT
fails, and the failure surfaces as
`Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected`
— which names the model and not the toolchain. The `thor:gpu0` staging recipe
two sections above already installs it for the same reason.

**`RC_LEASE_ID` DOES NOT EXIST on this fleet.** The leased worker carries
`RC_DEVICE`, `RC_JOB_ID` and `RC_TOKEN`. A script that defaults a lease id from
`$RC_LEASE_ID` therefore reads empty and refuses, which is what
`scripts/dflash2-speed-gate.sh` did when the `## Owed` O26 recipe was run
verbatim. Take `$RC_JOB_ID`. Read the variable off the worker rather than out of
a document, including this one: the fleet is not ours and the names can change.

### The `flock` orphan hazard that motivated the replacement

The harness family in this repository puts the `flock` handle on a **subshell**,
not on the `timeout` or wrapper process a reader would check. Kill the wrapper
and an ORPHAN survives holding the mutex, with its output pipe severed, and it
looks perfectly idle to every instrument a reader reaches for.

Measured 2026-08-17 (`.agents/specs/mtp-k-gt-1.md`, "What held the mutex"):
`nvidia-smi --query-compute-apps` was EMPTY and loadavg stayed near 1.1, which
reads as a finished holder. The holder was PID 333128, `bash -s 8000`, `PPid: 1`,
holding `fd 3` on the lock file, with `fd 1` and `fd 2` still pointing at the
pipe its dead `tee` had been reading. It was not idle. It held a live container
and was inside a readiness poll, and it blocked its own owner's restart for about
50 minutes as well as the queued gate. Read the whole process chain and
`/proc/<pid>/fd` before you call a lock stale, and never kill an unowned PID.

### Two ways a `strix:gpu0` job dies before it measures anything, 13 September 2026

Both cost a lease on that date and neither names itself in the failure.

**Ubuntu noble's own ROCm packages outrank `repo.radeon.com`, and
`rocm-hip-sdk` is unsatisfiable until you pin.** The `strix` worker is a bare
Ubuntu 24.04 container: no `/opt/rocm`, no `rocprofv3`, no `podman`, `apt-get`
and `/dev/kfd` present, and `repo.radeon.com` reachable. Adding the ROCm
repository is not enough, because noble ships `rocminfo` 5.7.1, `rocm-cmake`
6.0.0 and `hipcc` 5.7.1 in universe and apt prefers them:

```text
rocm-hip-runtime : Depends: rocminfo (= 1.0.0.70204-93~24.04)
                   but 5.7.1-3build1 is to be installed
```

The install then fails with "you have held broken packages", which names
neither ROCm nor Ubuntu. Pin the vendor repository above the distribution:

```sh
printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
  > /etc/apt/preferences.d/rocm-radeon-first
```

With the pin, `rocm-hip-sdk rocprofiler-sdk hsa-amd-aqlprofile libdw1t64`
installs in **431 s** and `rocprofv3` runs. `rocprofiler-sdk` alone was already
satisfiable without the pin, so a job that only profiles will not see this and a
job that also builds will.

**`exec >"$LOG" 2>&1` in a job script gets the job REAPED by `rc`.** Redirecting
the script's own stdout to a file closes the job's stdout, and `rc` treats that
as the end of the process:

```text
rc: log stream ended: unexpected EOF
rc: job b4e2330a-...: stragglers reaped after exit
```

Lease `b4e2330a` died that way about one second in, before it installed
anything, and `rc ps` showed no job. This is worth stating because the idiom is
in committed job scripts here, paired with a background loop that copies the log
to `/workspace`, and it reads as the careful thing to do. Stream to `rc`'s own
stdout instead and let `rc` keep the log; copy to `/workspace` at the end,
because `rc` logs age out within a day. Process substitution is worse, not
better: `exec > >(tee ...)` wedges the lease at exit.

### `rocprofv3` writes NOTHING on `strix:gpu0` unless you move its temp directory, 13 September 2026

**`rocprofv3` spills its trace records to `$CWD/.rocprofv3/<ppid>-<pid>-<domain>.dat`
and an `rc` job starts with `cwd = /`. From `/` that spill never comes back, so
the profiler aborts at output generation and writes a zero-row file.** Set
`ROCPROF_TMPDIR` to a directory under `/tmp`, or `cd` into one, and the same
command writes a complete trace.

The failure names nothing that points at a directory:

```text
E output_stream.cpp:111] Opened result file: .../12328_kernel_trace.csv
ring_buffer: munmap failed: Invalid argument
F ring_buffer.cpp:106] mmap failed with errno 22 :: Invalid argument
    ... abort
W tool.cpp:3104] [rocprofv3_error_signal_handler] rocprofv3 caught signal 6...
```

`strace` gives the argument the message omits: the call is
`mmap(NULL, 0, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)` and
**the length is zero**. `ring_buffer::load` (`ring_buffer.cpp:229` at
rocprofiler-sdk `97f5574fe`) reads the buffer size out of the spill file and
calls `init(_size)` without checking the read; when the read delivers nothing,
`_size` stays 0, `init` rounds 0 up to 0, and `mmap` of length 0 is `EINVAL`.
`init` then calls `destroy()` on the failed mapping, which is the `munmap`
warning printed one line earlier, and logs at FATAL, which aborts. The tool's
own `rocprofv3_error_signal_handler` re-enters on signal 6 and does not return;
on one lease it survived `SIGTERM` for 29 minutes and needed `SIGKILL`.

The default is in `output_config.hpp:81`: `tmp_directory = output_path` and
`output_path = "%cwd%"`. `--output-directory` moves the RESULT files only.
`rocprofv3`'s CLI never sets `ROCPROF_TMPDIR` (`source/bin/rocprofv3.py`), so
the spill follows the working directory whatever you pass.

Measured on one lease, a 50-dispatch HIP program, four arms in this order:

| Arm | `cwd` | `ROCPROF_TMPDIR` | kernel_trace.csv |
|---|---|---|---:|
| B1 | `/` | `/tmp/rpt` | 6602 B, **51 rows** |
| A1 | `/` | unset | 0 B, **0 rows**, `mmap failed` |
| C1 | `/tmp/cdtest` | unset | 6602 B, **51 rows** |

B1 ran FIRST and passed, so this is not a cold-start or an ordering effect: the
working directory is the whole variable. `/` is writable on the worker -- an
8 MiB `dd` to `/.rocprofv3/` succeeds at 2.9 GB/s -- so "the root filesystem is
read-only" is NOT the explanation, and the precise reason a spill under `/` is
unreadable is UNVERIFIED. The remedy does not depend on it.

Two corollaries. The recipe is:

```sh
mkdir -p /tmp/rpt
cd /tmp && ROCPROF_TMPDIR=/tmp/rpt timeout -s KILL <n> \
    rocprofv3 --kernel-trace --output-format csv --output-directory <dir> -- <cmd>
```

And **set `ulimit -c 0` around any `rocprofv3` invocation on this box.** The
abort dumps core into `cwd`, and a HIP process maps the whole VRAM carve: one
such core left **11.7 GB** in `/` on 13 September 2026.

Three hypotheses were tested and are FALSIFIED, so nobody needs to repeat them.
`ulimit -l` is 8192 on the worker but `ulimit -l unlimited` is permitted there
and the failing `mmap` is anonymous and unlocked, so the locked-memory limit is
not involved. `Seccomp: 0` and `CapEff: 000001ffffffffff`, so no filter and no
dropped capability. `rocprofiler-sdk` 1.1.0, `rocprofiler-register`,
`rocprofiler-sdk-rocpd`, `rocprofiler-sdk-roctx` and `hsa-amd-aqlprofile` are
all installed, so no package is missing. The 2026-09-07 capture's `podman` image
is not required; a bare worker writes the same trace once the spill has a
usable directory.

Two more ways to lose a lease to this tool, both measured the same day. **Never
pipe `rocprofv3` into `head`**: `head` closes the pipe, the tool's signal
handler catches the `SIGPIPE` and wedges, and the arm that should take 40 s
holds the box until `--max-runtime`. Redirect to a file and grep the file.
**Never run `find /` on this worker**: `/workspace` is a 7.3 TB CIFS mount and
the traversal does not come back.

## Registering your own environment

The profiles below are per-developer facts, not requirements: nothing here is
usable unless your own setup provides it. To make your hardware a gate
environment:

1. Copy `.env.example` to the untracked `.env` at the repository root and fill
   in what your setup has (reference checkouts, oracle, gate host, GPU lock,
   device arch and toolchain). Empty means unavailable, and the gates that
   need it stay `PENDING` for you.
2. Copy `developer-preferences.example.md` to the untracked
   `developer-preferences.md` for the policy choices (Git integration, which
   remote hosts you may use, contention policy).
3. Set `CHECKPOINT_ROOT` in your `.env` if you have shared or network storage,
   and download model weights there rather than onto a box's system disk. A
   30B bf16 checkpoint is ~60 GB; a build tree is ~169 GiB on its own, and a
   full disk surfaces as unrelated test failures rather than an obvious disk
   error. Fetching once to shared storage means every host, worktree and agent
   reuses it instead of each pulling its own copy. It states an INTENT and
   nothing more: no code in the tree reads `CHECKPOINT_ROOT`, so it neither
   redirects a download nor resolves a bare directory name — you place the
   weights under it and pass the full path onward. A setup with no shared mount
   leaves it empty and uses whatever the tool defaults to (usually the Hugging
   Face cache under `$HOME`).

   Two rules travel with it. Pin an explicit revision when you fetch:
   publishers re-quantize in place under an unchanged repo name, so a bare
   branch name is not reproducible and a checkpoint you gated against can
   change under you. And setting the variable authorizes nothing on its own —
   a large asset download still needs authority for the task.

4. Add a profile entry to this file, in the same shape as the entries below:
   hardware, arch, toolkit versions, oracle availability, and the box's
   quirks. A PR for it is welcome, so the shared record says where each gate
   can run. New accelerator classes (an AMD/ROCm box, an Intel GPU) register
   the same way and become the gate environment for their backend rows.

- **Rich local development/GPU profile (re-verified 2026-07-25):** NVIDIA GeForce RTX
  5070 Ti, 16 GiB, compute capability 12.0 (`sm_120`), driver 595.71.05. The
  cached `Qwen/Qwen3.5-4B` snapshot is the only model large enough for the
  local direct-load performance diagnostic; it cannot run the 27B/35B gates.
  The CUDA Nix shell must put `/run/opengl-driver/lib` before toolkit stubs.
  `flake.nix` now does so directly and no longer emits a malformed literal
  `LD_LIBRARY_PATH` expansion. A clean `nix develop .#cuda` reports
  `torch.cuda.is_available() == True` and Triton
  `GPUTarget(backend='cuda', arch=120, warp_size=32)` without a manual
  override. Build the current diagnostic with CMake CUDA arch `120a`, Triton
  vendored target `sm_120`, CUTLASS and FlashAttention-2 enabled.
- **CPU development path:** CPU reference backend + engine logic + CI
  development.
- **Ettore DGX release-gate profile**: device `dgx:gpu0`, host `dgx.casa` (claim
  it with `rc`, and read "Reaching a GPU" earlier in this file before you use the
  `ssh` recipes recorded here). DGX Spark, GB10 (Blackwell, **sm_121**),
  ~119 GB unified memory, 20 cores, CUDA toolkit 13.0.88 (nvcc); compute
  capability 12.1 → sm_121. Unified memory: both gate models fit
  in bf16; the machine is memory-bandwidth-bound (~273 GB/s class) — decode
  parity is a bandwidth/launch-overhead game, hence CUDA graphs + fused
  kernels in T0. Give each active claim its own `~/work/<claim>/` directory;
  never share a build tree between agents.
  - Non-interactive SSH does not put nvcc on PATH — prepend
    `export PATH=/usr/local/cuda/bin:$PATH` in remote build commands.
  - **The NAS mounts at `/usr/local/nas_share`, and `/mnt/nas_share` is GONE
    (re-verified 2026-08-16).** `.env` sets
    `CHECKPOINT_ROOT=/usr/local/nas_share/checkpoints`, where 18 checkpoint
    directories resolve, `nemotron-3.5-lightning-30b-nvfp4` and
    `nemotron-3.5-lightning-30b-gguf` among them. **Do not restore the old path
    as a convenience symlink.** `/mnt` is on the EPHEMERAL root overlay of this
    immutable Kairos OS, so anything created there is gone after the next
    reboot; `/usr/local` is `COS_PERSISTENT` and survives. That is the same
    property that made an earlier `/oem` `rootfs`-stage change cost a boot (see
    [[kairos-oem-rw-paths-change-cost-a-boot]]). Measured 2026-08-16, after the
    box returned from an 8 h 19 min outage: the mount itself came back because
    the `/oem` boot-stage unit worked and `findmnt /usr/local/nas_share` was
    clean, and `/mnt/nas_share` did not come back. Every path built on `/mnt`
    broke while `.env` still declared it, which blocks a checkpoint-loading gate
    silently — a gate that reads a path `.env` does not declare is not the gate
    its spec names. Check `findmnt /usr/local/nas_share` before you conclude
    that a checkpoint is missing (#1073).
  - **MANDATORY gate-build flags on this box (re-proven 2026-07-29).** A model
    gate configured WITHOUT `-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0` and
    `-DVLLM_CPP_TRITON=ON` is NOT the production stack: cutlass-off silently
    disables the sm120a NVFP4 fp4×fp4 GEMM *and* FlashAttention-2, triton-off
    swaps the vendored Triton-AOT GDN kernels for the hand kernels. On the 27B
    that flips the documented tok6 near-tie and turns the SACRED
    `test_qwen27_paged_engine` red (234/235 or 233/235 vs 235/235) with the
    source untouched — measured three-arm from ONE tree at main `d4492c03`. Hard-
    verify the configure log prints `CUTLASS found … enabling sm120a NVFP4
    cutlass GEMM`, `FlashAttention-2 prefill/decode: ENABLED`, and the
    `Triton AOT: … <- vendored … sm_121a` lines before trusting any gate or A/B.
    Never read the arch from `CMakeCache.txt` (`CMAKE_CUDA_ARCHITECTURES` there
    legitimately reads `75`, the `enable_language(CUDA)` probe default shadowed
    in `CMakeLists.txt`); read `VLLM_CPP_CUDA_ARCHITECTURES`, `flags.make`, or
    `cuobjdump -lelf`. This defect has voided work three times (2026-07-16 ×2,
    2026-07-28); the 27B gate now refuses to run without both flags.
  - Oracle venv: **PIN ADVANCED 2026-07-26** (see
    [specs/pin-advance.md](specs/pin-advance.md)). `~/venvs/vllm-oracle` is now a
    canonical symlink to the from-source **`~/venvs/vllm-oracle-next`** — the new
    active stack is **vLLM 0.26.0.dev0+g5559679** (source `55596792`, built for
    sm_121a: the exact commit has NO aarch64 wheel, so the oracle is a ~1.3 h
    from-source build not a pip install), **Transformers 5.14.1, Torch
    2.13.0+cu130, FlashInfer 0.6.15.post1, CUTLASS DSL 4.6.0, Triton 3.7.1,
    torchvision 0.28.0**. This advance unblocks DFlash (vllm#40898 mixed-attn fix,
    under `VLLM_USE_V2_MODEL_RUNNER=1`), Gemma-4 (`transformers.models.gemma4`),
    and OLMo-3 (nested rope). It was validated by the W0–W4 pin-advance re-gate:
    **zero real golden drift** (27B-W4A4 + 32B-NVFP4A16 bit-identical, 35B/Coder
    byte-stable — the W0-W2 "27B drift" was a capture-config near-tie, not the
    oracle), full `ctest` 296/299 GREEN on GB10 (the 3 fails pre-exist on main,
    unrelated). **ROLLBACK (immediately restorable):** `ln -sfn
    ~/venvs/vllm-oracle-v0.25.0-stage ~/venvs/vllm-oracle` restores the prior pip
    vLLM 0.25.0 stack (FlashInfer 0.6.13, Torch 2.11.0+cu130, CUTLASS DSL 4.5.2,
    Transformers 5.13.1, Ninja 1.13.0; install/serving SHA-256 `ab786eee…c297` /
    `536385d8…f506`, vLLM/Ninja `ec6d76ff…96c` / `abf71487…10b`, freeze
    `cf1636cc…fa5f`); the v0.24.0 dir remains at
    `~/venvs/vllm-oracle-v0.24.0-retired`. The §2D mechanical re-sync of
    upstream-changed mirrored files (rmsnorm-fusion #46998, ReplaySSM #48018,
    MoeWNA16 #44120, olmo3.py, …) is a DEFERRED follow-on (gate-correctness does
    not depend on it — goldens are bit-identical).
  - The only dependency-check exception is NVIDIA's
    `nvidia-cusparselt-cu13==0.8.0` wheel: PyPI served the aarch64 wheel
    (`sha256:400c6ed1…77c`), its library is an AArch64 ELF and direct
    `ctypes.CDLL`/Torch imports pass, but its internal WHEEL tag is
    `manylinux2014_sbsa`, so `pip check` reports it unsupported. This is a
    recorded vendor-tag defect, not silently treated as a green `pip check`.
  - Lock-held production-graph validation on the exact 27B snapshot passed both
    offline generation (16 input IDs, one output ID) and the actual text-only
    server: `/health` 200 plus `/v1/completions` 200 with exact 1+1 usage and
    `finish_reason=length`. Server log/response SHA-256 are
    `f56be69a…3787` / `82307db4…8e1` under
    `~/work/vllm-oracle-v0.25.0-stage-validation/2026-07-12-server-smoke`.
    The smoke rate is non-binding. Its first offline inference emitted one
    causal-conv Triton JIT warning, which remains a warmup/trace audit item.
    Online-gate manifests hash pandas package/distribution files plus Ninja and
    reject missing/drifted dependencies before the GPU lock; profiler launches
    prepend the venv `bin` to spawned EngineCore `PATH`.
  - **Run the CUDA `ctest` suite with `-j 1`.** GB10 memory is UNIFIED, so a
    `gpu_memory_utilization` reservation is HOST RAM: concurrent model gates
    stack into the same ~119 GB and the kernel starts killing. Measured
    2026-08-09 on a default-ON Triton build, `ctest -j 4` **OOM-rebooted this
    box** (`NVRM ... Out of memory [NV_ERR_NO_MEMORY]`), which is why the
    parallel-flake advice in the Apple/Metal profile below does not transfer
    here. Serialising also means every other probe queues behind the suite, so
    run attribution arms BEFORE a full suite, never during one.
  - **★ TREAT A SERVING LEG AS AT RISK OF A REBOOT TOO, NOT ONLY A BUILD OR A
    LOAD — THOUGH ONLY THE REBOOT IS OBSERVED AND ITS PLACEMENT NEXT TO THE LEG
    IS DERIVED — AND FROM INSIDE A LEASE `boot_id` IS THE ONLY INSTRUMENT THAT
    SEES A REBOOT AT ALL
    (measured 2026-08-19, [#915](https://github.com/mudler/vllm.cpp/issues/915)).**
    **Read the heading at the strength of its parts.** What `boot_id` observes is
    that a reboot happened somewhere between two readings about **11.2 hours**
    apart, and across the same unpinned clock boundary: the old value is last
    recorded in `clock-vllm-c1-r3.json`, written 10:18:51.7Z on `dgx`, and the
    new one landed in a log written 21:29:35.6Z on the local host. The serving
    leg occupied about
    **6.5 minutes** of that span. Putting the reboot next to the leg needs the
    DERIVED `/proc/uptime` bound below, and tying it to the worker's death is not
    claimed at all. So "under a serving load" is the working assumption this
    bullet is written for, not a measured fact.
    The pinned oracle's c8 denominator leg for `Qwen/Qwen3.8-27B` bf16 — vLLM's
    production graphed shape at `--gpu-memory-utilization 0.85
    --max-num-batched-tokens 8192` — answered `GET /health 200 OK` at 10:25:07Z
    with about 9,950 MB of `MemAvailable`, read 6,261 MB at 10:25:26Z, and the
    worker was then lost inside one 2-second sample, during the untimed warmup.
    That is OBSERVED, and it is the loss rather than the reboot. That
    configuration leaves roughly **6-7 GB of headroom** on this box. The
    earlier reboots this file records for this machine are a `ctest -j 4` and an
    oracle LOAD; this one is only ASSOCIATED with a server that was already
    healthy and serving-ready, and on that association "survived startup" is not
    a safe state.
    **Read `boot_id` in every leased job that loads anything large.** A lease
    gives you a pod, not the box's history: `uptime` resetting and
    `journalctl --list-boots` are host instruments a pod does not have, and a
    lost worker looks identical whether the pod died or the machine did. That
    ambiguity stood unresolved in the campaign record for a day.
    `/proc/sys/kernel/random/boot_id` is kernel-wide and regenerated per boot,
    so a changed value is a reboot and nothing else can forge it: a later job
    (`97cf3e63-e4a4-4506-bde7-f19f19be3bbf`) read
    `64c495a3-8c9c-4b20-8496-a97efda0e332` against the benchmark's
    `3fd9745a-d25a-426c-ba3c-97c958a85515`. **Do not promote a boot TIME derived
    from `/proc/uptime` to the same strength.** Read inside a pod, `/proc/uptime`
    is the host's only if the worker does not virtualize `/proc` — `lxcfs` does,
    and it cannot touch `boot_id` — so the identity change is observed and any
    derived timestamp carries that assumption. State it beside the number. The
    bound is also ONE-SIDED: the log mtime you subtract from is when the LAST
    line landed, so it is an upper bound on the read instant and never a
    midpoint. Here that gives a boot at or before 10:41:47.6Z with no lower
    bound, and the mtime is on a different host's clock from the uptime, with the
    offset unmeasured.
    **And a sampling watchdog cannot guard the REBOOT CLASS of failure at all.**
    A userspace sampler dies with the kernel, so there is no floor and no cadence
    at which it reports a reboot; that holds on its own and needs no link to any
    particular worker loss. Here the 2-second sampler never even saw a value
    below its own 5,000 MB floor. Detail in
    [`specs/qwen38-27b-bf16-gate.md`](specs/qwen38-27b-bf16-gate.md).
  - **GPU mutex:** this runs INSIDE an `rc` lease, never instead of one. The
    lease decides who gets the box. The mutex serialises the work of whoever
    holds it. Every CUDA test/model/serve/benchmark/profile holds the
    `${GPU_LOCK}` file mutex — **`$HOME/gpu.lock`**, which is what `.env.example`
    ships and what every script here falls back to via
    `${GPU_LOCK:-$HOME/gpu.lock}` — for the whole job or multi-arm series WHEN
    other agents may run GPU work concurrently (sole owner verified idle via
    `nvidia-smi` may skip). Mechanism: run GPU work as
    `flock ${GPU_LOCK:-$HOME/gpu.lock} -c '<command>'`, or take the lock once
    around an entire benchmark series so arms are never interleaved; waiting on
    the lock is normal, stealing it is not. Compilation, source inspection and
    file transfer do not need the lock. Never kill an unowned PID.
    **Check your `.env` before you measure anything:** a `GPU_LOCK` naming any
    other path takes a mutex nobody else holds, and `flock` succeeds on it, so
    the run is unserialised and only looks like someone else misbehaving. That
    cost a whole Marlin series (#777); an existing `.env` predating that fix
    must be repaired by hand.
  - Disk cleanup 2026-07-10 reclaimed ~368 GB from unrelated cached model sets,
    April-era autoresearch logits/F16-GGUF cache artifacts, the vLLM compile
    cache and stale rebuildable CUDA build trees. Active latency/PR workspaces,
    gate checkpoints, APEX GGUF evidence and sources were preserved; the volume
    had 359 GB free afterward. Maintain at least 200 GB headroom before adding
    competitor images.
- **Ettore Jetson Thor profile (sm_110 CUDA runtime gate)**: device `thor:gpu0`,
  host `192.168.68.23` (claim it with `rc` first)
  — NVIDIA Jetson Thor (Blackwell, **sm_110**), aarch64, 14 CPU cores, ~122 GB
  UNIFIED memory. `nvidia-smi --query-gpu=compute_cap` returns **11.0**. Host of
  the first non-GB10 runtime proof (`CLAIM-CUDA-SM110-RUNTIME`, 2026-07-27).
  - **REIMAGED, re-verified 2026-08-11.** The box is hostname **`kairos-4db2`**
    (Ubuntu 24.04 under Kairos), driver **595.78**, and there is **NO host CUDA
    toolkit, no cmake, no nvcc, no huggingface_hub** — the JetPack R38 /
    `/usr/local/cuda-13.0` profile described here before is GONE. **The GPU is
    usable only from inside a container** (developer statement, confirmed on
    box), and the leased `rc` worker IS that container. There is no
    `local-ai-worker` container on the host; the worker-restore discipline below
    does not apply in this state.
  - **The hand-run `sudo -n docker` recipe that used to sit here is DELETED, not
    demoted to an alternative, because this FILE ALREADY FORBIDS IT.** The
    section that opens this document — "Reaching a GPU: claim a lease, never
    `ssh`" — says to claim `thor:gpu0` with `rc run` or `rc hold` and never to
    `ssh` to a GPU box to run work. The Thor profile then taught the opposite in
    four places, so the document contradicted its own header, and a reader who
    scrolled to the profile for a recipe was taught to break the rule they had
    read twenty lines in. Bringing this section into line with its own file is
    the whole of the change; `rc` is not a new local convention.

    What the deleted recipe said, kept as history because it was measured and
    true on 2026-08-11, and because a future host-side operator may need it:
    `docker` needed `sudo` (the user is in group `admin`, not `docker`); the run
    needed `--runtime=nvidia` and NOT `--gpus all`, because the hook refuses the
    latter outright; and it needed `-e NVIDIA_DISABLE_REQUIRE=1`, because the
    image's `NVIDIA_REQUIRE_CUDA` enumerates bounded driver ranges topping out
    at `driver<576` while this box runs 595.78, which is newer and forward
    compatible. None of that is a procedure to follow now. Take `thor:gpu0` with
    `rc run`; the section below is the whole recipe.
  - **"Every sm_110 fast-path cell resolves EMPTY" was wrong, and it mattered.**
    Configure on `[110]` DISABLES `fp4-mma`, `cutlass-nvfp4`,
    `cutlass-nvfp4-sm100`, `cutlass-fp8`, `scaledmm-c3x-sm90`,
    `scaledmm-c3x-sm100` and `fa2`, and prints
    `CUDA feature marlin-nvfp4: ENABLED for [110]`. Read live at `944d7d947` on
    2026-08-22 and **re-read unchanged at `6756f9131` on 2026-08-23**
    (`/mnt/nas_share/rc/thor-w05-955/out/configure.log:16-23`, all eight cells in
    one place). Seven of eight is not eight, and the difference is the whole of
    [#962](https://github.com/mudler/vllm.cpp/issues/962): a Marlin NVFP4 kernel
    that disagrees with itself across block sizes ON THIS ARCH. The wrong line
    makes a live kernel defect read as an absent feature, which is exactly
    backwards. `.agents/backend-matrix.md` has said "FOUR of the five fast-path
    cells resolve EMPTY; `marlin-nvfp4` resolves `ENABLED for [110]`" since
    2026-08-11, so this line was already contradicted by another record.
  - `nsys` from `nsight-systems-cli` in the `nvidia/cuda:13.0.1` image was
    **2024.2.3 and could not trace CUDA here** ("does not contain CUDA trace
    data"). That was a property of that image. In the leased worker `nsys` is
    ABSENT altogether (measured 2026-08-22), so a graph or kernel-count
    measurement on Thor still needs one installed, and still needs a newer one.
  - **★ BUILD AND TEST IT INSIDE A LEASE — `rc run -d thor:gpu0`**
    (`dots3-note` W0.5, issue
    [#699](https://github.com/mudler/vllm.cpp/issues/699)). One job configures,
    builds CUDA-ON for sm_110, runs kernels on the device and produces the
    `ctest` baseline. Copy it rather than re-deriving it.

    **This REPLACES the `ssh` + `sudo -n docker build` + `sudo -n docker run`
    recipe W0.5 landed on 2026-08-15, and replaces it rather than joining it.**
    Two things were wrong with that recipe, and only the first is a rule.

    It reached a fleet device outside its lease. `thor:gpu0` is in the fleet, so
    a job that arrives by `ssh` makes `rc` report the box FREE while a build is
    on it — and this file's own opening section already said not to do that. The
    contradiction was internal, not a matter of taste: a reader who took the
    header seriously and a reader who scrolled to this profile for a recipe were
    given opposite instructions.

    And the digest-pinned image was never needed — though NOT for the reason an
    earlier draft of this section gave. It claimed the leased worker already
    supplies everything the image did, **and for the CUDA toolkit that is
    FALSE**; see "The toolkit is not in the image" below. The image is
    unnecessary because a job installs the toolkit itself in one step, which is
    what `dgx:gpu0` jobs already do
    ([#1213](https://github.com/mudler/vllm.cpp/issues/1213)), not because the
    toolkit is waiting there.

    What the worker DOES supply, measured 2026-08-22 in `rc run -d thor:gpu0` job
    `55810add-082e-461b-828b-b7cfe4dbb645` (log and artifacts under
    `/mnt/nas_share/rc/thor-w05-repair/`, which the worker sees as
    `/workspace/thor-w05-repair/`):

    | | |
    |---|---|
    | user | `uid=0(root) gid=0(root)` in the k3s pod `rc-worker-<id>`, 14 CPUs, aarch64 |
    | present | `bash sh git curl wget gcc/g++ 13.3.0 make cmake 3.28.3 ninja 1.11.1 pkg-config python3 3.12.3 pip flock` — this list is exactly what was probed, and `readelf`/`objdump` were NOT among them |
    | `nvcc` | **NOT part of the image. Install it.** Both of this lane's jobs happened to find `/usr/local/cuda-13.0` and nvcc 13.0.88 already there, and that was another job's leftover — see below |
    | ABSENT | **`shellcheck`**, **`cuobjdump`**, **`nsys`**, `sudo`, `docker` — and `cuobjdump` stays absent after the `PATH` prepend, which matters below |
    | `nvidia-smi` | plain, no `sudo`, **exit 0 with ZERO bytes on stderr**, reporting `NVIDIA Thor`, driver 595.78, `compute_cap 11.0` |
    | `/workspace` | `//192.168.68.102/Data`, 7.3 T, CIFS `file_mode=0664 nounix` — the SAME folder `dgx` and `orin` see, and `/mnt/nas_share/rc` on the devbox. `rc` copies nothing for you |
    | `/tmp` | the worker's own overlay, 918 G — but it was **94% used with 58 G free** on 2026-08-22. Read "Disk is shared" below before you build |
    | swap | **30.7 G of `zram`** (`/dev/zram0`, PRIO 100), with `vm.overcommit_memory=1`. Compressed RAM, not a backing store — see the reboot warning below |
    | reuse | **the container is REUSED between jobs**, so `/tmp` carries other jobs' trees and your own from last week |

    **★ The toolkit is not in the image, and BOTH of this lane's jobs were
    fooled by leftover state.** The `rc describe thor:gpu0` usage sheet says it
    plainly: *"There is no CUDA toolkit (no `nvcc`). The driver is injected and
    `nvidia-smi` works, but compiling CUDA needs the toolkit — install it, or
    keep the build on the host."* The worker container is LONG-LIVED, so one
    job's `apt-get` is still sitting there for the next job, and the
    `leasing-a-gpu` skill warns about exactly that leak.

    The chain is visible inside this file. The 2026-08-19 job ran in pod
    `rc-worker-hqfj4` and found nvcc 13.0.88 already present. The pod was then
    recreated — the `#1380` note further down records that the NEXT worker,
    `rc-worker-m4d7t`, had lost `/tmp` **and `/usr/local/cuda-13.0`, "so the job
    had to `apt-get` the toolkit again"**. The 2026-08-22 job then ran in
    `rc-worker-m4d7t` and found the toolkit present. It was present because that
    other job had just installed it. Neither observation is a property of the
    image, and the pod has since been recreated again by `worker_lost`, so
    neither is true now.

    **So install it as a step and assert the postcondition.** Never write a
    recipe whose first requirement is that somebody else ran a job on the same
    pod first. This is the same shape as the `shellcheck` defect this section
    already documents: a recipe that works only because of undocumented state on
    the box, which the next reader cannot reproduce.

    **Probe it with `command -v` and an explicit else-branch, never with
    `tool --version | tail -1`.** `nvcc --version 2>/dev/null | tail -1 || echo absent`
    prints nothing and exits 0 when `nvcc` is missing, because `tail` succeeds on
    empty input and the `||` never fires. That reads as "present but quiet" and
    it nearly put a false line in this table.

    **1. Stage the tree where the worker can see it.** `rc` copies nothing.

    ```sh
    # WHERE THE SHARE IS MOUNTED DEPENDS ON THE HOST YOU TYPE THIS ON.
    # On the devbox it is /mnt/nas_share/rc; on the dgx HOST that path is GONE
    # and the share is /usr/local/nas_share/rc (see the DGX profile). Resolve it,
    # never assume it -- staging to a stale path silently stages nothing, and the
    # job then fails for a reason that looks unrelated.
    for c in /mnt/nas_share/rc /usr/local/nas_share/rc; do [ -d "$c" ] && NAS=$c && break; done
    test -n "${NAS:-}" || { echo "FATAL: no NAS mount on this host"; exit 96; }
    D=$NAS/<your-dir>                       # the worker always sees this as /workspace/<your-dir>
    mkdir -p "$D"
    git rev-parse HEAD > "$D/BASE_SHA"      # a baseline with no SHA beside it is not a baseline
    git archive --format=tar HEAD | gzip -1 > "$D/src.tar.gz"
    ```

    A `git clone` inside the worker also works — it has `git` and egress, and
    other rows use it — but only for a branch you have pushed. The archive is how
    you test a tree that is not on the remote.

    **2. Submit one bounded job, and NAME the device.**

    ```sh
    rc run -d thor:gpu0 --max-runtime 120m \
      -- bash /workspace/<your-dir>/run.sh      # the job prints a heartbeat; see below
    ```

    Name it rather than selecting on `class=train`, which also matches `dgx:gpu0`
    and `orin:gpu0` and neither of those is sm_110; `gpu_model=NVIDIA-Thor` is
    the only selector that means this box. Naming queues you behind whoever holds
    it, which is the lease working. Never select on `vram` — it reads `[N/A]`
    here and matches nothing.

    **`--idle-timeout` counts the job's OWN stdout, and a build that logs to a
    file is silent for its whole duration.** A `cmake --build` redirected into a
    log prints nothing for twenty minutes, and the idle killer does not care why.

    **Print a heartbeat. That is the remedy that was actually proven here, and
    `--idle-timeout 0` is NOT a second way to do the same thing.**
    `rc run --help` reads
    `--idle-timeout duration   kill the job if it produces no output for this long (0 = device default)`,
    so zero selects the DEVICE DEFAULT rather than disabling the kill. An earlier
    draft of this section prescribed `--idle-timeout 0` as the fix, which would
    hand a future agent a long quiet build and a false belief that it is
    protected. A one-line background loop is the whole of it:

    ```sh
    ( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S)"; done ) &
    HB=$!
    # ... the quiet work ...
    kill "$HB" 2>/dev/null; wait "$HB" 2>/dev/null
    ```

    The margin this closes is not theoretical. The 2026-08-19 run built for
    22 min 44 s in silence against `--idle-timeout 25m` and survived with about
    two minutes to spare. **`--max-runtime` is the bound that matters and it is
    not optional**; the heartbeat is what stops the idle killer from firing
    inside it.

    **3. What the job does.**

    **This script is the whole recipe, including the disk discipline the prose
    below mandates.** An earlier draft printed a shorter version and left the
    disk rules to the prose; "copy it rather than re-deriving it" means the
    script is what actually gets run, so a guard that lives only in a paragraph
    is a guard nobody executes.

    ```sh
    #!/bin/bash
    set -u
    SRC=/tmp/src
    # NEED_GB is no longer a guess, and it is deliberately NOT set to the
    # measured figure. A finished tree + build-cuda is 25 GiB (`du -sh /tmp/src`
    # at 6756f9131, 2026-08-23; /tmp went 116 G -> 92 G across that job). 60
    # keeps a ~2.4x margin for the other lanes' trees that share this
    # reused overlay, which is the thing that actually varies. The other data
    # points: a build that COMPLETED with 154 G free (2026-08-19), one that did
    # NOT complete with 58 G free (2026-08-22, worker lost, so disk is a
    # hypothesis rather than a proven cause), and one that COMPLETED with 116 G
    # free (2026-08-23).
    NEED_GB=${NEED_GB:-60}
    free_gb() { df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'; }

    # --- clean up on ANY exit, including the kill. A job that dies holding its
    # --- tree is what took this box out of the pool.
    cleanup() { rm -rf "$SRC"; kill "${HB:-}" 2>/dev/null; wait "${HB:-}" 2>/dev/null; }
    trap cleanup EXIT INT TERM

    ( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S) disk=$(free_gb)G"; done ) &
    HB=$!

    # --- DISK, before anything else. The container is REUSED, so other jobs'
    # --- trees are still here and the free space is shared with them.
    df -h /tmp
    du -sh /tmp/* 2>/dev/null | sort -rh | head -10
    rm -rf /tmp/src /tmp/thor-w05-src*          # reclaim this lane's old trees, not just mine
    if [ "$(free_gb)" -lt "$NEED_GB" ]; then
      echo "REFUSING: /tmp has $(free_gb) GiB free, below the NEED_GB=${NEED_GB} floor."
      echo "The floor is MEASURED, not a placeholder: a finished tree + build-cuda"
      echo "is 25 GiB (6756f9131, 2026-08-23). 60 is a deliberate ~2.4x margin for"
      echo "the other lanes' trees that share this REUSED overlay -- that is what"
      echo "varies, not the build. Reclaim space before lowering NEED_GB."
      echo "A CUDA build that runs out of space fails as unrelated compile errors."
      exit 95
    fi

    # --- The CUDA toolkit is NOT in the worker image. Install it UNCONDITIONALLY.
    # --- `command -v nvcc` is NOT a sufficient precondition: a partial leftover
    # --- puts nvcc on PATH while `include`/`lib64` are missing, and CMake then
    # --- dies with "Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS
    # --- CUDA_CUDART_LIBRARY)", which names the version and denies the toolkit
    # --- in one line. apt-get is idempotent, so paying it every run costs
    # --- seconds and removes the whole class.
    # `cuda-toolkit-13-0` is NOT in Ubuntu's own archive. Add NVIDIA's repo
    # first, or the install fails to resolve on a freshly recreated pod -- and
    # succeeds on a pod where some earlier job already added it, which is the
    # same leftover-state trap in a second guise.
    apt-get update -qq
    apt-get install -y -qq wget ca-certificates gnupg
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
    dpkg -i /tmp/ck.deb
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-13-0
    export PATH=/usr/local/cuda/bin:$PATH
    CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

    # --- Assert the POSTCONDITION the build actually needs, not the binary.
    command -v nvcc >/dev/null   || { echo "FATAL: no nvcc after install"; exit 90; }
    test -f "$CUDA_HOME/include/cuda_runtime.h" \
      || { echo "FATAL: nvcc present but no cuda_runtime.h under $CUDA_HOME"; exit 90; }
    ls "$CUDA_HOME"/targets/*/lib/libcudart.so* >/dev/null 2>&1 \
      || ls "$CUDA_HOME"/lib64/libcudart.so*    >/dev/null 2>&1 \
      || { echo "FATAL: no libcudart under $CUDA_HOME"; exit 90; }
    nvcc --version | tail -2                    # record WHICH toolkit built this

    # --- the cubin reader, installed unconditionally and asserted with an
    # --- explicit else-branch. An absent cuobjdump piped into `grep -o` looks
    # --- exactly like a clean empty histogram; that is how 2026-08-19 recorded
    # --- an empty one without noticing.
    apt-get install -y -qq cuda-cuobjdump-13-0
    if command -v cuobjdump >/dev/null; then CUOBJ=1; else
      CUOBJ=0; echo "### cuobjdump ABSENT after install -- cubin proof stays OWED"
    fi

    mkdir -p "$SRC"
    tar -xzf /workspace/<your-dir>/src.tar.gz -C "$SRC"
    test -f "$SRC/CMakeLists.txt" || { echo "FATAL: untar"; exit 92; }

    cmake -S "$SRC" -B "$SRC/build-cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF
    cmake --build "$SRC/build-cuda" -j 4

    # --- PROVE it is a CUDA build before you believe any test result.
    ldd "$SRC/build-cuda/libvllm.so" | grep -Ei 'cudart|cublas'
    find "$SRC/build-cuda" -name '*.cu.o' | wc -l          # the DENOMINATOR
    if [ "$CUOBJ" -eq 1 ]; then
      for o in $(find "$SRC/build-cuda" -name '*.cu.o'); do
        echo "== $o"; cuobjdump --list-elf "$o"
      done > /tmp/cubin.log 2>&1
      grep -o 'sm_[0-9]*' /tmp/cubin.log | sort | uniq -c  # expect N sm_110
      echo "objects scanned: $(grep -c '^== ' /tmp/cubin.log)"
    fi
    ( cd "$SRC/build-cuda" && ./tests/test_cuda_backend )

    ( cd "$SRC/build-cuda" && ctest -j1 --timeout 1800 --output-on-failure )

    # --- MEASURE what this actually cost. 25 GiB at 6756f9131; re-read it
    # --- rather than trusting that, because the tree grows.
    du -sh "$SRC" "$SRC/build-cuda"
    echo "### /tmp free at end: $(free_gb) GiB"
    # cleanup() removes $SRC on the way out, on success AND on the kill path.
    ```

    **`cuda-toolkit-13-0` is the package, and a bare `cuda-nvcc-13-0` is NOT a
    usable fallback.** An earlier draft of this script fell back to it when the
    full toolkit failed, and that fallback is worse than no fallback: it
    satisfies `command -v nvcc`, so the guard passes, and then
    `CMakeLists.txt:1717` runs `find_package(CUDAToolkit REQUIRED)` and `:1719`
    links `CUDA::cudart CUDA::cublasLt`, neither of which a compiler-only package
    supplies. The build dies at configure with the same misleading
    missing-CUDA line as a partial leftover. This build's own `ldd` resolves
    `libcudart.so.13` and `libcublasLt.so.13`, so those are requirements and not
    conveniences. If you must install a minimal set rather than the metapackage,
    it is `cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcublas-dev-13-0` — and the
    postcondition assertions above are what tell you whether you got it right.

    **The `sbsa` path in that keyring URL is the aarch64 one**, and it is what
    both this box and `dgx` need; an `x86_64` URL resolves and then installs
    nothing usable here. The concurrent `gate-fp8-thorbuild` job took the same
    route on 2026-08-22 and its configure reported
    `The CUDA compiler identification is NVIDIA 13.0.88` at
    `/usr/local/cuda-13.0/bin/nvcc`, which is independent corroboration of both
    the package and the version.

    **cuRAND is NOT among the requirements**, in case a reader reaches for it:
    `grep -rn curand src include CMakeLists.txt cmake` returns nothing, so this
    tree never uses it. The `curand` mention elsewhere in this file belongs to
    DFlash2's flashinfer JIT on `dgx`, which is a different lane.

    Build in `/tmp` or `/root`, never on `/workspace`: the share is CIFS with
    `nounix`, so it stores no symlink and presents no exec bit, and `chmod +x`
    there fails with `Operation not permitted`. Write every environment repair
    UNCONDITIONALLY and assert its postcondition, because the container is reused
    and a conditional repair is skipped on the second job and then reports
    success. Copy what you want to keep back to `/workspace`.

    **★ Disk is shared, and getting this wrong took the box out of the pool.**
    A run on 2026-08-22 wiped only its own tree, left an earlier tree of this
    same lane in place, and built on an overlay that was already 94% used with
    58 G free. Partway through the build `thor:gpu0` went
    `unhealthy (no contact)` with `out of the pool: worker_lost`, and the job
    died at SIGTERM. Host memory was NOT the cause — the sampler read 116 GiB
    available one minute before contact was lost, so this is not the
    overcommit-collapse signature below. **`worker_lost` did not self-heal
    QUICKLY, and it did eventually clear.** The device still read
    `unhealthy (no contact 2h38m)` two and a half hours later; by 2026-08-22 it
    was `ready` again with `disk_free_bytes` back to about 123 GiB. Do not clear
    a quarantined device yourself — that needs an admin token and is a human's
    call — and do plan for hours rather than minutes, because there is no second
    sm_110 device to fall back to and an sm_110 gate simply stops meanwhile. **Report free space, reclaim every tree
    the lane has left behind rather than only your own, refuse rather than build
    into a nearly full disk, and remove your tree when you finish.** A CUDA build
    that runs out of space surfaces as unrelated compile errors, which is the
    [[enospc-makes-checkers-emit-false-policy-refusals]] shape.

    No CUTLASS directory is passed, and that is correct rather than lazy: every
    CUTLASS-dependent cell is arch-gated off for `[110]`, so supplying CUTLASS
    changes nothing here. `-DVLLM_CPP_TRITON=OFF` is deliberate — the vendored
    Triton-AOT path has never been built for sm_110, and turning it on is a
    separate measurement rather than a default.

    **4. Prove the build is really a CUDA build, because a silent CPU fallback
    is the failure mode this exercise exists to catch.** Configure at
    `6756f9131` on 2026-08-23 prints `CUDA target architectures: 110` and
    resolves `/usr/local/cuda/bin/nvcc` (NVIDIA 13.0.88), and that build's
    `ldd build-cuda/libvllm.so` resolves `libcudart.so.13` and
    `libcublasLt.so.13` out of `/usr/local/cuda-13.0/targets/sbsa-linux/lib`,
    over **33** `*.cu.o` objects. It read 32 objects at `0764ded2b`; the count
    moves with the tree and is not a constant to assert.

    **The cubin check is no longer owed. It was RUN, with the reader present and
    asserted, and it reads 33 objects and 33 `sm_110` cubins — one each.**
    Measured 2026-08-23 at `6756f9131`
    (`/mnt/nas_share/rc/thor-w05-955/out/cubin.log`, histogram in
    `cubin-histogram.txt`): `cuobjdump --list-elf` over every `*.cu.o` yields
    `33 sm_110` and nothing else, over `33` objects scanned. That retires the
    unverified 2026-08-15 claim of "30 objects, one `sm_110` cubin each" — the
    shape was right and the count was stale.

    **Keep the two guards that made it trustworthy, because the absent-reader
    trap is what cost the earlier readings.** `cuobjdump` was recorded ABSENT on
    2026-08-22 even after prepending `/usr/local/cuda/bin`, and an absent tool
    piped into `grep -o` with stderr discarded looks exactly like a clean empty
    result — which is how a run on 2026-08-19 recorded an empty histogram
    without noticing. So: **`apt-get install -y cuda-cuobjdump-13-0`
    unconditionally, `command -v cuobjdump` with an explicit else-branch, and
    COUNT the objects you scanned beside the histogram.** A histogram with no
    denominator cannot tell "33 of 33" from "33 of 300".
    **What is still unresolved is WHY it was absent before.** This job found it
    at `/usr/local/cuda/bin/cuobjdump` after an `apt-get` that produced no output
    under `-qq`, on a pod whose CUDA toolkit was already installed by an earlier
    job, so whether the explicit install supplied it or the leftover toolkit did
    cannot be separated from this log. The step is what makes it reproducible
    either way; do not read this run as evidence that the metapackage always
    carries it.

    **5. Runtime proof, on the device.** At `6756f9131`,
    `./tests/test_cuda_backend` reported `CUDA compute capability: sm_110`,
    `pageable=1 integrated=1 UnifiedMemory=true` and
    `DeviceMemoryIsHostAddressable=false`, **7/7 cases, 26/26 assertions,
    exit 0** (`out/test_cuda_backend.log`). It read 6/6 and 25/25 at
    `0764ded2b`; the suite grew a case.

    **`UnifiedMemory=true` with `DeviceMemoryIsHostAddressable=false` is the
    pair that matters, and it is the axis two of this baseline's entries turn
    on** — see the FP8 rows in the table below. Unified memory does not imply
    the host can dereference a device pointer, and on this box it cannot.

    A hand-written `nvcc -arch=sm_110` kernel also compiles and runs in the
    worker: `kernel_returned=1234 err=no error`, exit 0 (measured `0764ded2b`,
    not re-run here).

    **6. Parallelism, and what is actually measured.** Build at **`-j 4`**, run
    `ctest` at **`-j 1`**.

    `ctest -j1` is not caution; it is the same rule the GB10 profile above
    states, for the same reason. Memory here is UNIFIED, so a model gate's
    `gpu_memory_utilization` reservation is HOST RAM, concurrent gates stack into
    the same ~122 GB, and this box reboots rather than OOM-kills. Serial costs
    **632.35 s** for 598 tests at `6756f9131` (it was 419.97 s for 553 at
    `0764ded2b`), so there is nothing to buy by stacking them, and
    it means the shipped baseline IS a `-j1` reading rather than a `-j4` reading
    with a separate `-j1` re-run asserted beside it.

    **The two numbers the earlier record used to justify `-j4` are corrected
    here.** It claimed "all 14 reproduced at `-j1`" and "peak host memory ~6 GB
    of 122 GB". The `-j1` claim's artifact
    ([#955](https://github.com/mudler/vllm.cpp/issues/955) cites
    `/home/mudler/thor-w05/ctest-j1-rerun.log`) is on the Thor HOST, which a
    lease cannot read, so it can no longer be checked; the ~6 GB figure had no
    artifact at all. Re-measured with a 2-second `MemAvailable` sampler beside
    the job:

    | Phase | `MemAvailable` min | max | own footprint |
    |---|---|---|---|
    | build `-j4`, 1364 s (`0764ded2b`) | 114.3 GiB | 117.8 GiB | ~3.5 GiB |
    | `ctest -j1`, 420 s (`0764ded2b`) | 111.2 GiB | 118.6 GiB | ~7.4 GiB |
    | whole job, build + `ctest`, 2338 s (`6756f9131`) | 111.4 GiB | 118.8 GiB | ~7.4 GiB |

    The `6756f9131` row is one sampler across both phases rather than two, so it
    bounds the run without splitting it; its min matches the earlier `ctest`
    min to a tenth of a GiB, which is the corroboration that matters. Artifact
    `/mnt/nas_share/rc/thor-w05-955/out/memsample.txt`.

    Neither phase goes near the wall, so `-j4` for the build is cheap and now has
    an artifact.

    **Disk cost is measured now, so `NEED_GB` need not stay a guess.** At
    `6756f9131` the extracted tree plus a completed `build-cuda` is **25 GiB**
    (`du -sh /tmp/src`), and `/tmp` went from **116 GiB free to 92 GiB** across
    the job — a 24 GiB delta that agrees with the `du`. The `NEED_GB=60` floor
    in the script is therefore conservative by better than a factor of two, and
    the honest thing is to say so rather than to lower it silently: 60 leaves
    room for the other lanes' trees that share this overlay, and it refused
    nothing on a box that had 116 GiB. **That is a statement about a BUILD and a unit suite and it
    transfers to nothing else** — a model load here is what took the box down
    three times, and the whole-tree build that correlates with a later
    `unknown (no contact)` on this device is
    [#1380](https://github.com/mudler/vllm.cpp/issues/1380), recorded further down this same profile.

    **7. `nvidia-smi`, corrected.** The earlier record said it "dies"
    unprivileged with `NvRmMemInitNvmap failed: error Permission denied`, works
    only under `sudo -n`, and reads `[N/A]` in its memory columns. Three
    corrections, with different provenance, so read which is which.

    **Measured here, in the lease:** it runs with no `sudo`, **exit 0 and ZERO
    bytes on stderr** — not one `NvRm` line — reporting `NVIDIA Thor`, driver
    595.78, CUDA 13.2. Inside a leased worker there is no privilege problem to
    work around at all.

    **Reported by the developer from the HOST, and deliberately NOT re-measured
    here, because doing so needs the `ssh` this recipe refuses to take:** an
    unprivileged host call also exits 0 and prints the table, with the `NvRm`
    lines going to STDERR beside a successful report. On that reading, "it
    refuses" was stderr being taken for a verdict.

    **The memory field has two spellings, both in the same job's output:** the
    table column reads `Not Supported`, while
    `--query-gpu=memory.total,memory.used --format=csv` reads `[N/A]`. Either way
    this is an integrated GPU with no separate VRAM counters, the same shape as
    [[gb10-has-no-dram-counters-ncu-memory-pct-is-a-lie]]. Do not build a
    memory-headroom check on `nvidia-smi` here; read `free -g` or
    `/proc/meminfo`, which is what the sampler above does.

    **The `flock ${GPU_LOCK}` wrapper this section used to prescribe is gone
    with the `ssh` recipe.** On a fleet device the LEASE is the mutex — `rc`
    hands the whole device to one job. The file mutex is for a GPU that is not a
    fleet device, and taking one over `ssh` while another session holds the box
    through `rc` is precisely the two-mutex collision recorded at the top of this
    file, which already retained a speed axis as VOID.

    **8. `ctest` BASELINE — `6756f9131`, 2026-08-23, `ctest -j1 --timeout 1800
    --output-on-failure` inside an `rc run -d thor:gpu0` lease, 632.35 s wall:**

    ```text
    598 tests: 573 passed | 3 skipped | 22 FAILED
    ```

    **Artifacts, because this section indicts earlier records for citing none.**
    Job `8bf39567-9334-4f7e-aa27-43a2aa867bb7` on `thor:gpu0`, pod
    `rc-worker-kk96r`, written to `/mnt/nas_share/rc/thor-w05-955/out/` (the
    worker sees it as `/workspace/thor-w05-955/out/`): `ctest-j1.log` carries
    every failure with `--output-on-failure`, `failures.txt` is the
    `(name, mode)` list, `ctest-N.log` the 598-test enumeration, `build.log`
    (`-j4`, 1643 s, exit 0), `configure.log`, `ldd.log`, `cu_o_count.txt`,
    `cubin.log` + `cubin-histogram.txt` (the cubin proof), `test_cuda_backend.log`,
    `memsample.txt` (2 s `MemAvailable`), and `ctest-shellcheck.log` — the
    single-test re-run after installing `shellcheck`, which is a CONTROL here
    rather than a fix; see below. `run.sh` is the exact script that produced all
    of it and `rc-job-stdout.log` the whole job's stdout, both copied in beside
    the outputs so the recipe and the readings cannot drift apart. `BASE_SHA`
    and `src.tar.gz` in the parent directory are the
    staged input. **This is scratch on a shared NAS and may be reaped**, which is
    a reason to quote the numbers here as well as the paths, not a reason to omit
    the paths.

    The previous baseline was `0764ded2b`, 2026-08-19, **553 tests, 534 passed /
    3 skipped / 16 red**, 419.97 s, job `1b2512f0-0a43-44cb-b4a4-b54c22b59bd9`
    under `/mnt/nas_share/rc/thor-w05-repair/out/`. Its numbers are kept here
    only as the far side of the diff below.

    **The 22 FAILED tests in the table below are the sm_110 baseline, and are NOT
    to be "fixed" by a row that merely builds here.**

    Separately, three tests are Skipped for an absent checkpoint, unchanged
    across both runs: `test_modelopt_mixed_precision_checkpoint`,
    `test_voxtral_e2e`, `test_qwen35_paged_engine`. They are not part of the 22,
    and `Skipped` ranks BELOW `Failed` — see the mode ranking below.

    **★ ZERO SEGFAULTS. Every one of the 22 is mode `Failed`.** All three crashes
    the previous baseline recorded are gone, and none of them was replaced by a
    worse mode elsewhere. That is the single largest movement this lane has seen,
    and the mechanism is named below.

    **The gate is `(name, failure mode)` PAIRS. A row regresses on Thor when it
    adds a NAME, or when it changes a recorded MODE FOR THE WORSE.** Counting
    names is provably too
    weak, and this baseline's own history is the proof: between `5a0ffe9e3` and
    `2daa3287f` five tests went `Failed` → `SEGFAULT` with no name change, and
    the list lengthened by one only because the same upstream change also shipped
    a new test file — so a name-counting gate would have scored
    [#960](https://github.com/mudler/vllm.cpp/issues/960) GREEN on five of the
    six crashes it introduced. The Mode column is part of the baseline, not
    decoration, and it costs no extra measurement because `ctest` prints the mode
    beside every failure. Recorded as
    [#955](https://github.com/mudler/vllm.cpp/issues/955), the sm_110
    counterpart of [#907](https://github.com/mudler/vllm.cpp/issues/907).

    **Direction matters, and an earlier draft of this gate left it out.** Two
    kinds of movement are IMPROVEMENTS and must not be scored as regressions.
    A name LEAVING the list is one. A mode getting BETTER is the other, and it
    is not hypothetical: this run saw three of them.

    Rank the modes, worst first: **`SEGFAULT` / `Subprocess aborted` / `Timeout`,
    then `Failed`, then `Skipped` / `Not Run`, then PASSING.** A crash becoming a
    clean assertion failure is progress, because a crash takes the rest of its
    binary's cases with it — at `0764ded2b` `test_capi` reported `61 skipped`
    behind its SIGSEGV, and at `6756f9131` the same binary runs all 66 cases.
    Only movement DOWN that ranking, or a new name, is a regression. Record every
    move in either direction; score only the bad ones.

    **`Skipped` sits BELOW `Failed` on that list and is NOT an improvement, which
    is the trap in ranking at all.** Three tests already skip on this host for an
    absent checkpoint, so the mode is live in this very baseline: a red test that
    starts skipping has stopped being measured, not started passing, and a
    checkpoint that quietly goes missing looks exactly like a fix. **So a name
    that LEAVES the failure list is only an improvement when you have seen it
    PASS.** No name left the list this time, so nothing needed that check here —
    but the previous run did need it and did it, confirming `test_ops_fp8_cpu`
    and `test_ops_fused_chain` `Passed` in the same log rather than assuming.
    A departure you cannot show green is an unexplained change, and it goes in
    the report as one.

    **The table is keyed on NAME, never on the `ctest` ordinal.** Ordinals move
    whenever a test file is added, and they moved between every pair of runs this
    lane has taken. **The first-failing-assertion LINE NUMBERS move too** — this
    run alone moved `test_platform` `:307`→`:420` and
    `test_gguf_device_fit_reach` `:278`→`:463` with the assertion unchanged — so
    read the line as a pointer to re-derive, not as part of the key.

    | Test | Mode | First failing assertion | Cause |
    |---|---|---|---|
    | `test_serve_low_tools` | Failed | `tests/tools/test_dflash2_speed_harness.py` — 1 error + 3 failures of 517, `FAILED (failures=3, errors=1, skipped=1)` | **THE CAUSE CHANGED.** No longer the absent `shellcheck`; see the note under the table. Unattributed, [#1802](https://github.com/mudler/vllm.cpp/issues/1802) |
    | `test_platform` | Failed | `:413` `CHECK(cu.supports_fa2_attention())` false, then `:420` `CHECK(cu.is_device_capability_family(120))` false | the TEST hardcodes the sm_12x family and an FA-2 build. Thor is 11.0 with `fa2` DISABLED. Two failing assertions now, one before |
    | `test_gguf_device_fit_reach` | Failed | `:463` `CHECK(Backend().queues_destroyed == destroyed_before + 1)` → `0 == 1`, case "device fit: the AUTO arm refuses when the accelerator queue CAN be created" | red and UNATTRIBUTED since 2026-08-15; folded into [#1802](https://github.com/mudler/vllm.cpp/issues/1802) rather than left to be re-noticed a fourth time |
    | `test_linear_method` | Failed | `:247` `after == before + 1` → `0 == 1`, case "MXFP4 fused gate_up … fused path ran" | the MXFP4 fused path does not run on sm_110. Also red on GB10 ([#907](https://github.com/mudler/vllm.cpp/issues/907)) |
    | `test_qwen3_5_gdn_spec_routing` | Failed | `:530` `CHECK(bad == 0)`, case "GDN merged FP8 qkvz == the two split fp8 GEMMs, bitwise", 8 failing assertions | unchanged from `0764ded2b`. Also red on GB10 ([#907](https://github.com/mudler/vllm.cpp/issues/907)) |
    | `test_qwen3_5_gdn_spec_routing_glue_fuse_off` | Failed | same assertion, same case | same |
    | `test_qwen3_5_gdn_spec_routing_fused_chain_off` | Failed | same assertion, same case | same |
    | `test_deepseek_v2_forward` | Failed | `:559` THREW `cuda mla_prefill_attention: built without the vendored FlashAttention-2` | no FA-2 on sm_110 |
    | `test_capi` | Failed | `:953` `CHECK((text == "1" \|\| text == "2"))`, case "capi: structured_choice constrains greedy decoding", and `:975` the streaming sibling. `66 cases \| 64 passed \| 2 failed \| 0 skipped` | **MODE IMPROVED from `SEGFAULT`.** The `:487` ABI-v8 SIGSEGV is gone and the whole binary now runs — the `61 skipped` [#994](https://github.com/mudler/vllm.cpp/issues/994) complained of is 0. What remains is two structured-output cases, which are NOT the recorded defect |
    | `test_cuda_ops` | Failed | `:106` `CHECK(bad == 0)` → `6 == 0` and `7 == 0`, case "CUDA silu_and_mul matches CPU" | **NEW NAME.** Also red on GB10 ([#907](https://github.com/mudler/vllm.cpp/issues/907)) at 439/440, against 438/440 here. Tracked in [#1802](https://github.com/mudler/vllm.cpp/issues/1802) |
    | `test_backend_cross_device` | Failed | `:2063` `CHECK(got == ref_b)` ("MoeSiluMul matches the CPU oracle within NMSE <= 5e-4") and `:2601` `CHECK(dout.Download() == ref_c)`. 80205/80207 assertions | **NEW NAME**, [#1802](https://github.com/mudler/vllm.cpp/issues/1802). Elements differ in the last digit |
    | `test_llama_embedding_fold` | Failed | `:254` `CHECK(engine[i] == doctest::Approx(direct[i]).epsilon(1e-5))` | **NEW NAME**, [#1802](https://github.com/mudler/vllm.cpp/issues/1802) |
    | `test_mtp_depth` | Failed | `:738` `CHECK(st.capture_shapes == 0)`, 103/104 assertions | **NEW NAME**, [#1802](https://github.com/mudler/vllm.cpp/issues/1802) |
    | `test_qwen3_dflash2_draft` | Failed | `:2574` `CHECK(r.generate_threw.empty())`, 352/353 assertions | **NEW NAME**, [#1802](https://github.com/mudler/vllm.cpp/issues/1802) |
    | `test_ops_attention_dense_fa2` | Failed | `:692` `CHECK(Mismatches(on, ref) > 0)` → `0 > 0`, case "attention-dense-fa2 VT_FA2_DENSE=0 restores the scalar kernel at hd-128" | **NEW NAME**, [#1802](https://github.com/mudler/vllm.cpp/issues/1802). Reads like a test arch-assumption rather than a kernel defect: the case asserts the knob-ON path DIFFERS from the scalar reference, and with `fa2` DISABLED for `[110]` they are the same kernel |
    | `test_ops_fp8_cutlass` | Failed | `:191` and `:207` THREW `vt: no kernel for op MatmulFp8Cutlass (id 54) on device cuda (type 1), and the portable CPU reference tier is NOT eligible: this backend does not report its device memory host-addressable` | **MODE IMPROVED from `SEGFAULT`, and this is the answer to [#1725](https://github.com/mudler/vllm.cpp/issues/1725)'s first half.** The fall-through is gone; the op refuses loudly. See below |
    | `test_ops_matmul_fp8_block_cuda` | Failed | `:521-523` `CHECK(what.find("N is 576"))`, `("multiple of 128")`, `("sm120")` all `npos`, and `:529` `CHECK(after.refused - before.refused == 1)` → `0 == 1`, case "G2 upstream's CUTLASS case is refused by name…". `5 cases \| 1 passed \| 4 failed` | **MODE IMPROVED from `SEGFAULT`**, but [#1725](https://github.com/mudler/vllm.cpp/issues/1725) is NOT closed by it: the refusal is the GENERIC provider one, not `BlockFp8Runnable`'s by-name refusal, so neither the message nor the `refused` counter is what the test asserts |
    | `test_ops_moe_grouped` | Failed | `:1262` `CHECK(bitdiff == 0)`, logged at `:1260` as `NVFP4 block8-vs-block16 M=8 K=4096 N=4096 bitdiff=15/32768` | **the one substantive standing sm_110 finding, [#962](https://github.com/mudler/vllm.cpp/issues/962), and it REPRODUCED byte-identically** 176 commits later — same count, same shape, same line. `marlin-nvfp4` IS `ENABLED for [110]`, so this is a live kernel disagreeing with itself across block sizes, not an absent feature. Its MXFP4 sibling reads `bitdiff=0/32768` at `:1173` in the same run |
    | `test_ops_mla_prefill` | Failed | `:340` and `:437` THREW FA-2 absent | no FA-2 on sm_110 |
    | `test_ops_mla_chunked_context` | Failed | `:790` THREW FA-2 absent | same |
    | `test_mla_attention_block` | Failed | `:999` and `:1044` THREW FA-2 absent | same |
    | `test_op_parity` | Failed | `:2490` `output_cbor_sha256` mismatch TWICE in "qwen27 GDN BA BF16 projection matches vLLM 0.25 oracle (**dgx-only**, CUDA)" | dgx-captured goldens replayed on Thor; the case names itself dgx-only and runs anyway |

    **What moved between 2026-08-19 (`0764ded2b`, 553 tests / 16 red) and
    2026-08-23 (`6756f9131`, 598 tests / 22 red), 176 commits — 123 of them
    touching `src/`, `include/`, `tests/` or `CMakeLists.txt`, 384 files and
    +91,929 / -3,288 lines.** Expressed as the gate expresses it:

    | | |
    |---|---|
    | names ARRIVED | **6** — `test_cuda_ops`, `test_backend_cross_device`, `test_llama_embedding_fold`, `test_mtp_depth`, `test_qwen3_dflash2_draft`, `test_ops_attention_dense_fa2`, all `Failed`, all [#1802](https://github.com/mudler/vllm.cpp/issues/1802) |
    | names DEPARTED | **0** |
    | modes IMPROVED | **3** — `test_capi`, `test_ops_fp8_cutlass`, `test_ops_matmul_fp8_block_cuda`, all `SEGFAULT` → `Failed` |
    | modes WORSENED | **0** |
    | `Skipped` set | unchanged, the same 3 absent-checkpoint tests |

    A count of names reads 16 → 22 and calls that six regressions. The pairs read
    six arrivals, zero departures and three IMPROVEMENTS — including the
    disappearance of every crash on the box — which is what actually happened.

    **★ [#1725](https://github.com/mudler/vllm.cpp/issues/1725) MOVED, and the
    prediction this file recorded was right.** The previous entry said
    `cffe59b02` "rewrites the reference-tier dispatch … on the axis of unified
    memory, and Thor is a unified-memory box", so both FP8 SEGFAULT rows were
    "specifically suspect". They were. Neither op crashes now, and the exception
    text names the mechanism itself, in full and with its own `file:line`:

    ```text
    vt: no kernel for op MatmulFp8Cutlass (id 54) on device cuda (type 1), and the
    portable CPU reference tier is NOT eligible: this backend does not report its
    device memory host-addressable, so a host kernel may not dereference what it
    allocated (unified memory is true, which is a DIFFERENT property). Build a
    native kernel for this op or run it on the CPU device
    at src/vt/op_provider.cpp:563
    ```

    **Read the parenthesis.** The message distinguishes unified memory from
    host-addressability *by name*, which is precisely the confusion `cffe59b02`
    was written to remove, and Thor is the box where the two come apart:
    `test_cuda_backend` reports `UnifiedMemory=true` with
    `DeviceMemoryIsHostAddressable=false`. The tier is correctly refused
    instead of being handed device pointers to dereference. That is the outcome
    `src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu:56-58` asserts — an arch
    outside the cell "keeps refusing by name … the honest answer and not the
    #960/#844 fall-through" — and the source and the measurement now agree.

    **It is HALF resolved, not resolved, and the residue is precise.**
    `test_ops_fp8_cutlass` is down to the two cases that legitimately need the
    kernel. `test_ops_matmul_fp8_block_cuda` still fails 4 of 5 cases, because
    the refusal arrives through the GENERIC `op_provider` path rather than
    `dense_fp8_block::BlockFp8Runnable`'s by-name refusal: its G2 case asserts
    the message contains `N is 576`, `multiple of 128` and `sm120`, and that a
    `refused` counter increments, and none of that happens. #1725 was therefore
    RE-SCOPED rather than closed, on 2026-08-23, and its title now reads
    "`kMatmulFp8BlockScaled` refuses GENERICALLY instead of by name outside
    `VT_CUTLASS_FP8_ARCHS`". The original blind spot
    is unchanged: `cutlass-fp8` is ENABLED on GB10, so no CI lane can see either
    state.

    **★ These two never belonged to #960, and the attribution is kept here
    because it is provenance rather than superseded detail.**
    [#960](https://github.com/mudler/vllm.cpp/issues/960) was CLOSED COMPLETED on
    **2026-08-16** by `d607fec4c`, three days before the **2026-08-19**
    `0764ded2b` measurement, and a closed issue cannot own a live crash. **Get
    the direction of that timeline right**: the FIRST measurement, `2daa3287f` at
    2026-08-15 20:34Z, predates #960's own creation at 21:03Z the same day — the
    issue was filed BECAUSE of that run, so "closed before the first measurement"
    is exactly backwards and was corrected on 2026-08-23.

    **#960's fix was real, and this lane credits it.** `QuantFp8Static` moved
    into an unconditional TU, `test_ops_fp8_cpu` went GREEN, and the three
    `qwen3_5_gdn_spec_routing` tests improved `SEGFAULT` → `Failed` at
    `0764ded2b`. But `kMatmulFp8Cutlass` and `kMatmulFp8BlockScaled` are
    different ops, registered from TUs that `CMakeLists.txt:1790-1791` compiles
    only for `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a), so on sm_110 the same
    fall-through shape survived that fix on two more ops. `cffe59b02` is what
    finally closed it, and #1725 is what owns the residue. Do not read the
    surviving fall-through as evidence that #960 was ineffective.

    **★ [#962](https://github.com/mudler/vllm.cpp/issues/962) did NOT move, and
    "byte-identically" is the word that makes this evidence rather than a second
    sighting.** It reproduces at `bitdiff=15/32768` — same M/K/N, same count,
    same line — 176 commits and four days later, with the MXFP4 sibling clean at
    `0/32768` in the same binary.

    **A bit-exact repeat rules out the explanations a bare repeat does not.** It
    is not sampling noise, not a race, not a tolerance sitting near its edge, and
    not a property of one build's scheduling: any of those would move the count.
    A single observation of `15/32768` was one reading; two identical readings
    across that much churn make it a deterministic, corroborated defect on a path
    configure reports as `marlin-nvfp4: ENABLED for [110]`. It stays open, and it
    remains the only substantive standing sm_110 numerical finding.

    **The corollary for whoever fixes it:** the defect is reproducible on demand,
    so it does not need a hunt for conditions. Build for `[110]`, run
    `test_ops_moe_grouped`, and read `:1260`.

    **★ The `shellcheck` entry is STALE AS AN EXPLANATION, and the decision it
    justified has to be re-read.** This section used to say the canonical
    baseline does not install `shellcheck` and carries `test_serve_low_tools` as
    a named entry, because the failure was exactly
    [#961](https://github.com/mudler/vllm.cpp/issues/961) — an absent instrument
    reading as a code verdict — and the same job proved it by installing
    `shellcheck` and re-running that test alone to green.

    **That is no longer true.** `73ada0df8` (#1661/#1662) fixed the guard on
    `main`: `tests/tools/test_online_gate_startup.py:263-267` now catches
    `FileNotFoundError` and skips. The string `shellcheck` does not appear
    anywhere in this run's `ctest-j1.log`. The test still fails, for an unrelated
    reason — four cases in `tests/tools/test_dflash2_speed_harness.py`
    (`ShellDriverTest`), 1 error and 3 failures of 517. **The control was run
    again and it now falsifies the old conclusion instead of confirming it:**
    after `apt-get install -y shellcheck` (0.9.0), the same single test re-run
    reports `FAILED (failures=3, errors=1)` against the baseline's
    `FAILED (failures=3, errors=1, skipped=1)` (`ctest-shellcheck.log` versus
    `ctest-j1.log`) — **the same four cases, and the ONLY difference is the
    vanished skip.** Installing the instrument changes nothing about the failure.

    **That vanished `skipped=1` is itself the second proof, and it is worth
    claiming rather than glossing.** The one test that skipped in the baseline is
    the `shellcheck` guard; with the binary present it stopped skipping and
    PASSED. So the control does not merely fail to reproduce the old
    explanation — it independently demonstrates that the instrument was the only
    thing the install changed, and that it changed nothing about the four live
    failures.

    So: the entry stays in the list, the "do not install `shellcheck`"
    instruction stays (it costs nothing and #961 is armed on other hosts), and
    the CAUSE column is now [#1802](https://github.com/mudler/vllm.cpp/issues/1802)
    rather than #961. **#961 itself was CLOSED COMPLETED on 2026-08-23**, acting
    on this record's own prompt to check whether it still described anything
    live: `73ada0df8` had fixed its guard while referencing the sibling filing
    #1661/#1662, which left #961 orphaned rather than resolved. The close was
    verified against `origin/main` — the probe is wrapped in
    `except FileNotFoundError` with `skipTest` in both arms — with this run as
    independent corroboration that the failure mode is gone.
    This is the [[the-state-was-not-the-one-you-believed]] shape: an entry whose
    `(name, mode)` pair never moved while everything underneath it did.

    **★ The `-j1` re-run artifact question from the previous baseline is
    settled.** That record noted the older `-j1` claim cited
    `/home/mudler/thor-w05/ctest-j1-rerun.log`, on the Thor HOST, which a lease
    cannot read. This baseline is a `-j1` reading start to finish, taken inside
    the lease, with its log on the shared NAS. Nothing here depends on a
    host-side path any more.

    **Re-measure whenever the base SHA moves across `src/`, `include/`,
    `tests/` or `CMakeLists.txt`.** A stale baseline is worse than none, because
    the next agent reads a regression as the floor.

    Two things NOT to conclude from this table. The FP8 and FA-2 groups are
    **feature absence surfacing as a thrown exception**, which is the loud
    failure the seam is supposed to produce, and they are not sm_110 numerical
    bugs. And the FA-2 message reads *"MLA prefill on **sm_121** IS
    FlashAttention"* while running on sm_110: the text is hardcoded to the GB10
    arch, so do not read an arch out of it.

    **`shellcheck` is still ABSENT from the leased worker, and the standing
    instruction is unchanged: the canonical baseline does NOT install it.** What
    changed on 2026-08-23 is the REASON, and the paragraphs below are kept
    because a reader who remembers the old one needs to see it withdrawn rather
    than quietly replaced.

    **What was true until `73ada0df8`.**
    `tests/tools/test_online_gate_startup.py` shelled out to `shellcheck` and
    RAISED rather than skipping when the binary was absent, which is an absent
    instrument reading as a code verdict
    ([#961](https://github.com/mudler/vllm.cpp/issues/961)). The baseline
    therefore owned the entry rather than hiding it. In the deleted image the
    test passed, so the 2026-08-15 list read 15 rather than 16 — and the record
    explained that by saying "`shellcheck` is in the Dockerfile above", while the
    Dockerfile printed directly above it installed `cmake ninja-build git
    python3 python3-dev ca-certificates` and nothing else. The page designated
    the repo copy authoritative, so the authority contradicted the behaviour it
    was cited to explain. The 2026-08-19 job then proved the entry was exactly
    the instrument, by installing `shellcheck` 0.9.0 and re-running only that
    test: **`1/1 Passed, 27.88 s`**.

    **That proof is now WITHDRAWN, because the same control was rerun and came
    back the other way.** `73ada0df8` (#1661/#1662) fixed the guard —
    `test_online_gate_startup.py:263-267` catches `FileNotFoundError` and skips
    — so the instrument no longer reads as a verdict here at all, and the string
    `shellcheck` appears nowhere in the `6756f9131` `ctest-j1.log`.
    `test_serve_low_tools` is still red for an unrelated reason, and installing
    `shellcheck` 0.9.0 and re-running that test alone now reports
    `FAILED (failures=3, errors=1)` against the baseline's
    `FAILED (failures=3, errors=1, skipped=1)` (`out/ctest-shellcheck.log`) —
    the same four cases, differing only in the vanished skip, which is the
    guard itself passing once its binary is present. **Installing the instrument
    changes nothing about the failure.**

    **So keep the instruction and drop the justification.** Not installing
    `shellcheck` still costs nothing and still leaves #961's shape armed on hosts
    that have not taken the fix, but it is no longer what this entry measures.
    The entry's cause is [#1802](https://github.com/mudler/vllm.cpp/issues/1802).
    **The general lesson is the durable part:** the `(name, mode)` pair for this
    test did not move across 176 commits while its cause was replaced entirely,
    which is [[the-state-was-not-the-one-you-believed]] — a green-looking gate
    key over a changed world. Rerun a control; do not cite one.

    **One `rc` detail worth carrying.** A job whose `trap ... EXIT` leaves a
    background sampler running exits through
    `killed: stragglers reaped after exit`, with a non-zero client exit code
    AFTER `### ALL DONE` has already printed. Every phase result is on stdout by
    then. Do not read that line as a failed measurement; `kill` and `wait` on
    explicit PIDs if you want a clean code.
  - **★ THIS BOX REBOOTS INSTEAD OF OOM-KILLING — size every load for it.**
    `vm.overcommit_memory=1` ("always overcommit"): the kernel
    grants memory it cannot back, and touching those pages takes the WHOLE MACHINE
    down. Signature: container `exit=255`, `OOMKilled=false`, NO `dmesg`/journal OOM
    line, host reboot (`uptime` resets). Observed **three times on 2026-08-11**
    loading a 27B: bf16 (52 GB target + 8.2 GB draft) twice, and NVFP4 (25 GB +
    8.2 GB) once, the last of which left the box unreachable pending a power cycle.
    A 27B target alone loads and runs fine; it is target+draft that crosses the
    line, consistent with the documented transient double-hold in the load path
    (`114→67 GiB`, see [[gb10-nvfp4-load-recipe-context-first-shard-release]]).
    **Do not run a >25 GB model plus a second checkpoint here.** Measured OK on
    this box: Qwen3-4B bf16 spec-off warm ~24.6 tok/s; Qwen3.6-27B bf16 spec-off
    warm 4.42 tok/s (portable kernels, no fast paths — absolute numbers are NOT
    comparable to GB10).

    **The "zero swap" clause this warning used to carry is STALE as a reading of
    `free`, and the warning is UNCHANGED in force.** Measured 2026-08-22 inside a
    lease: `free -g` reads `Swap: 30 total, 0 used, 30 free`, and
    `/proc/sys/vm/overcommit_memory` still reads `1`. But `swapon --show` names
    what it is — **`/dev/zram0`, 30.7 G, PRIO 100** — and zram is a COMPRESSED
    BLOCK DEVICE BACKED BY RAM. It adds no storage behind the 122 GiB; it trades
    CPU for a compression ratio on pages that stay in the same physical memory,
    and its own pool grows in RAM under pressure. So the original clause was
    wrong about the literal reading and right about the thing it protected: there
    is no backing store here. **Do not read 30 GiB as headroom.**
    [#1363](https://github.com/mudler/vllm.cpp/issues/1363) still owes two
    questions: whether it changes the observed failure mode at all, and when the
    device appeared. Nobody has re-run the load that took the box down, and
    nobody should do so casually. Run a `MemAvailable` sampler beside any load
    here; `nvidia-smi` is blind to all of it.
  - **★ A BUILD CAN DO IT TOO, and the load list above is not the whole
    envelope. Measured 2026-08-19** ([#1380](https://github.com/mudler/vllm.cpp/issues/1380)).
    An `rc run` job on `thor:gpu0` ran `cmake --build <dir> -j 8` over the whole
    tree — **1098 targets**, which is a compile phase followed by a link phase
    that runs several `ld` processes at once, each mapping a multi-hundred-MB
    static image. Roughly five minutes in, `rc devices` went from `busy` to
    `unknown (no contact)` and then to `unhealthy`, the job vanished from
    `rc ps`, and its `/workspace` log stopped mid-phase with no error written.
    The box recovered on its own about twenty minutes later and now reads
    `ready`. **It was NOT re-leased while unhealthy** — AGENTS.md forbids
    clearing a quarantined device, and nothing was cleared.
    **"Most likely" is the honest register here.** The signature matches the
    reboot-instead-of-OOM shape this file already documents, and a parallel link
    of that tree is the largest concurrent host-memory demand in the build. What
    was NOT measured is the kernel's own account: nobody read `uptime`, `dmesg`
    or `journalctl --list-boots` on the host afterwards, so "the parallel link
    exhausted memory and the box rebooted" is inference from the fleet's view
    alone.
    **One piece of corroboration WAS measured, on the next job.** The worker that
    picked up the following lease reported a DIFFERENT pod name (`rc-worker-m4d7t`
    against `rc-worker-hqfj4` before), and its `/tmp` had lost the source tree and
    the build directory the earlier jobs had left there, and `/usr/local/cuda-13.0`
    was gone so the job had to `apt-get` the toolkit again. So the worker
    container was certainly recreated. That still does not distinguish a host
    reboot from a pod restart, which is exactly the gap the boot list closes.
    **What would settle it:** re-run the same full build at `-j 8` while
    sampling `free -m` from inside the job, and read `journalctl --list-boots` on
    the host afterwards — a GAP in the boot list is the evidence
    ([[kairos-oem-rw-paths-change-cost-a-boot]]), and a host that did not reboot
    would falsify it and point at the worker pod instead.
    **The instruction that already existed is the one to follow.** The build
    recipe above prescribes `cmake --build ... -j 4`. It used to say "`-j4` is
    deliberate"; that phrasing is gone, replaced by the measured sampler table
    under "Parallelism, and what is actually measured", and the instruction is
    unchanged. Every NAMED-TARGET build in that
    campaign — five jobs, `-j 8`, one to four targets each — completed without
    incident. It is the whole-tree build that correlates. Use `-j 4` for a full
    build, and prefer named targets.
    **This is the SECOND recorded `unknown (no contact)` for this device.** The
    first was 2026-08-17, cited in AGENTS.md as the reason a lost controller
    contact is a live state rather than a hypothetical. **Two readings do not
    make a rate, and neither one's cause was established**, so what this second
    occurrence adds is that the state recurs and must be planned for — not that
    the box has a property. Treating `unknown (no contact)` as a state a long
    job can enter is warranted by either occurrence alone. Whether the two share
    a cause is exactly what the boot list above would answer, and until somebody
    reads it, the count is a reason to keep looking rather than a finding.
  - `k3s` runs here and is `enabled`. **The `rc` worker is one of its pods**, so
    stopping `k3s` stops the lease mechanism itself and takes `thor:gpu0` out of
    the fleet. The old note that `sudo systemctl stop k3s` frees its pods (5
    containerd shims survive) is history from the host-toolchain era; do not run
    it. It was not the crash cause.
  - **Getting code and weights onto this box goes through the lease, not
    `ssh`.** Source: `git archive` to `/mnt/nas_share/rc/<dir>` — never rsync,
    see [[dgx-transfer-git-archive-not-rsync]] — or `git clone` inside the worker
    for a branch you have pushed. Weights: `/workspace` is the SAME NAS folder
    from `dgx` and from Thor, so a checkpoint placed there once is visible to
    both and no host-to-host copy is needed at all. The recipe this replaced
    moved weights dgx→Thor with `tar -ch | ssh ... tar -x`, which is the bypass;
    its footnote that the reimage changed the host key is history rather than an
    instruction.
  - **Oracle CAVEAT UPDATE (2026-08-12, `CLAIM-MM-SPEED-AUDIO-ENC-FA2`):** both venvs
    import cleanly now — `vllm-oracle-next` reports `0.23.1rc1.dev1511+g555967922`.
    That string is a **`setuptools_scm` nearest-ancestor-tag artefact, NOT an identity
    mismatch**: HEAD is `5559679229bc`, which IS the pin, and setuptools_scm names a
    dev build after the newest tag reachable from the commit, not after the release the
    commit belongs to. **Assert the oracle BY COMMIT, never by version string** — the
    recorded `0.26.0.dev0` string will not match and nothing is wrong. Its source tree
    `~/work/vllm-src-5559679` is present again (890 MB).
  - **`soundfile==0.14.0` is required in `~/venvs/vllm-oracle-next` for Voxtral
    (2026-08-12).** Without it the pin cannot tokenize the audio prompt at all, so the
    pinned oracle is not gateable for the Voxtral vehicle. Installed; recorded here and
    against issue #375. `~/venvs/vllm-oracle-v0.25.0-stage`
    (vLLM 0.25.0 + mistral_common 1.11.5 — the Voxtral golden-capture stack) **ran a full
    teacher-force to completion**, so the "crashes in EngineCore KV-cache/model init"
    below is at least partly a PATH artefact: in a non-login shell Triton's JIT dies
    `RuntimeError: Failed to find C compiler` AFTER the weights load, which surfaces as
    `EngineCore failed to start`. Export **`CC=/usr/bin/gcc`** alongside the documented
    `ninja` PATH fix and it runs. The `vllm-oracle` symlink still points at the 0.25.0
    rollback rather than the pin (issue #375, open).
  - **★ `CC=/usr/bin/gcc` is STALE for the reimaged host, and the correction is to
    run the oracle IN A CONTAINER (2026-08-17, `SPEC-MTP-K-GT-1`).** The bullet above
    is right about the failure and wrong about the cure on this host. Measured on
    `kairos-17dd`: there is no `gcc`, no `cc`, no `clang`, no `ninja` and no `nvcc`
    anywhere on the host, and `/usr/include` carries neither `stdio.h` nor
    `python3.12/Python.h`, so there are no glibc headers and no crt objects either.
    Exporting `CC=/usr/bin/gcc` therefore names a file that does not exist. Triton
    3.7.1 in the pinned venv ships only `ptxas`/`cuobjdump`/`nvdisasm`, no C
    compiler, so nothing in the venv supplies one.
    **What this looks like if you do not know it:** the weights load, the engine
    then dies `RuntimeError: Failed to find C compiler`, and vLLM reports
    `Engine core initialization failed. See root cause above. Failed core proc(s): {}`.
    That is an INSTRUMENT failure wearing the shape of a verdict about the model.
    Do NOT reach for `enforce_eager` to get past it: it is forbidden as a
    denominator, and it would silently change the thing being measured.
    **The cure**, and the shape `~/rs35b/run_oracle.sh` already used: run the host
    venv inside `nvidia/cuda:13.0.1-devel-ubuntu24.04` with
    `python3 python3-dev ninja-build build-essential libnuma1` installed, `-v
    $HOME:$HOME`, `CC=/usr/bin/gcc` and `/usr/local/cuda/bin` on `PATH`. The image
    ships python **3.12.3**, which matches the venv's `pyvenv.cfg` exactly, so the
    HOST venv resolves inside the container. Bake the toolchain into an image
    (`~/mtpgate/Dockerfile.oracle`, `mtpgate-oracle:1`) rather than `apt-get`ing it
    per leg: a leg that must reach the network to start can fail for a reason that
    has nothing to do with the measurement. Assert `gcc` and `ninja` INSIDE the
    container before the model loads, so a broken image aborts by name instead of
    four minutes later as an engine error. Container egress WAS available on
    2026-08-17; the box has been recorded without it before, which is the argument
    for baking rather than installing.
    **★ AND THE PINNED ORACLE CANNOT CURRENTLY LOAD A 27B HERE AT ALL: it eats the
    WHOLE MACHINE in the step after `torch.compile`, and `gpu_memory_utilization`
    does NOT control it.** Measured the same day, once the toolchain fix let an
    oracle get that far for the first time. At `gpu_memory_utilization=0.75` the
    engine held about **110 GiB of HOST RAM** while `nvidia-smi` reported only
    26 GiB on the device, hung 45 minutes at loadavg **260** with **0 GiB
    available**, and `sshd` stopped completing a banner exchange while the box
    still answered ICMP. Killing the container took it from 118 of 119 GiB used to
    4 of 119 in under ten seconds.
    **The obvious attribution to that 0.75 was tested and REFUTED.** A second run
    at **0.30**, with a 5-second host-memory sampler running, collapsed the same
    way: `avail_mb` 87683 at 09:00:47 and **0** at 09:02:25, loadavg 1.19 to 39.90.
    Weight loading finished with 66 GiB free and `torch.compile` finished with
    88 GiB free, so the collapse is neither of those. It is the step immediately
    AFTER compilation and it is insensitive to the KV-pool fraction, which points
    at the profiling forward and the graph capture (`max_num_batched_tokens=8192`,
    `cudagraph_capture_sizes: [1, 2, 4, 8]`, every allocation host-backed here).
    **That last part is a hypothesis with a located step, not a result.** Vary
    those one at a time with the sampler running and believe nothing without an
    A/B. **The 0.75 run THRASHED for 42 minutes and survived (`boot_id` and
    `uptime` unchanged); the 0.30 run REBOOTED THE BOX** — `boot_id` moved
    `5bbdc432…` to `bd5c6e7a…` and `journalctl --list-boots` shows boot `-1`
    ending 09:10:15Z against boot `0` beginning 09:13:55Z. So a lower fraction is
    NOT a safety margin: assume the box is at risk on every attempt. Always run a
    `MemAvailable` sampler beside any load here; `nvidia-smi` is blind to all of
    it, and the sampler is what turned a 45-minute mystery into a timestamped
    100-second collapse. While sshd was answering intermittently one connection
    returned `Permission denied (publickey)`; that is a memory-pressure artefact,
    not a credential problem, and the same key worked seconds after the reboot.
    **A cleanup trap is not a stop button.** Both DGX drivers used
    `trap cleanup EXIT INT TERM` where `cleanup` resets the clocks and RETURNS, so
    `SIGTERM` reset the clocks and the script then started its NEXT leg on a box
    with no memory left. Put an `exit` on the signal path, and `docker kill` the
    current named container inside the handler: `timeout` signals `docker run`, and
    the container outlives it.
  - **Oracle CAVEAT (2026-07-27):** the pinned vLLM oracle on dgx.casa was found
    DEGRADED — `~/venvs/vllm-oracle`→`vllm-oracle-next` (0.26.0.dev0) is an editable
    install whose source tree `~/work/vllm-src-5559679` was pruned (dangling; `import
    vllm` fails), and `~/venvs/vllm-oracle-v0.25.0-stage` (vLLM 0.25.0) now crashes in
    EngineCore KV-cache/model init. A fresh teacher-force could not be run this
    session; the sm_110 near-tie verdict rests on the COMMITTED gap-0 golden. Repair
    the oracle before the next gate that needs a fresh capture.
- **Ettore Apple/Metal profile**: `ssh 192.168.68.103` — Mac mini, Apple M4 (10 CPU
  cores), 16 GB unified memory, arm64, macOS 26.5.2. Use it for the MLX-backed
  `vt::` backend, Metal op parity, and small-model bring-up. It cannot hold the
  27B/35B gate models; gate-scale Apple performance needs a larger-memory Mac.
  **Re-verified 2026-07-22 (`CLAIM-BACKEND-FANOUT-1`), correcting the stale
  2026-07-10 line:** only the **Command Line Tools** are installed, NOT full
  Xcode — so the offline `metal` shader compiler is absent (`xcrun -sdk macosx
  metal` fails). That does **not** block MSL: runtime compilation via
  `newLibraryWithSource:` was verified working, together with a numerically
  correct dispatched compute kernel. **CMake IS already installed** (brew
  4.1.0 at `/opt/homebrew/bin`, missing from the non-interactive PATH — always
  `export PATH=/opt/homebrew/bin:$PATH` in remote commands); `ninja` is not
  (make works). **MLX is NOT installed** (`brew install mlx` -> 0.32.0, pulls
  `python@3.14`) and is not required for native-MSL bring-up. Device facts:
  `hasUnifiedMemory=YES`, `MTLGPUFamilyApple9` + `Metal3`, SIMD width 32,
  32 KiB threadgroup memory, 11.84 GiB recommended max working set, ~30 GiB
  free disk. Our tree configures AND builds there under AppleClang 21 with
  three Clang-only `-Werror` fixes, and 108,952 portable-tier assertions pass.
  **Updated 2026-07-22 (W0 landed):** the FULL tree (library + every test) now
  builds `-Werror`-clean on the M4 with the Metal backend ON, and the fix count
  is **seven**, not three — a full build surfaced four more than the spike's
  lib-only probe (see the fan-out spec § Work breakdown "W0 landed"). Configure
  with plain `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`: `VLLM_CPP_METAL`
  defaults to `AUTO` and turns itself ON for an Apple host with an ObjC++
  compiler. Add `-DVLLM_CPP_METAL=OFF` for a CPU-only A/B.

  **TWO PRE-EXISTING macOS TEST GAPS — expected failures, not regressions.**
  Both fail identically with `-DVLLM_CPP_METAL=OFF`, i.e. they are unrelated to
  Metal, and neither is fixed yet:
  - `test_serve_low_tools` — the Python bench tooling calls Linux-only
    `os.sched_getaffinity` (`tests/tools/test_gdn_packed_component.py`) and
    `POSIX_FADV_DONTNEED` (`tools/bench/drop_file_cache.py`).
  - `test_safetensors` — `MappingRssKb` reads `/proc/self/smaps`, which macOS
    does not provide, so it returns 0 and the RSS assertions cannot hold.

  `test_capi` and `test_openai_conformance` are ctest-PARALLELISM flakes on this
  box (and on Linux); they pass on rerun. Prefer `ctest -j 3` **here, on this
  16 GB Mac mini only** — it is not general advice, and in particular the DGX
  profile above requires `-j 1` because its unified memory OOM-reboots the box
  under a parallel CUDA suite.
  `test_engine_core_proc` used to fail under heavy parallel ctest in the case
  "EngineCoreProc: abort-mode shutdown aborts in-flight requests" (measured ~1/5
  in isolation under load, 2026-07-28). **FIXED 2026-09-01
  ([#2482](https://github.com/mudler/vllm.cpp/issues/2482)), and this page named
  the wrong cause and the wrong remedy until then.** It was never a race with the
  teardown: `InprocClient::shutdown()` JOINS the engine thread, so the abort frame
  is already on the queue by the time the scan starts. The scan was capped at 1000
  dequeues while the request carried `max_tokens=100000`, so it free-ran token
  deltas into an unbounded queue and, given more scheduler time under load, wrote
  more than 1000 of them ahead of the abort frame. The failing run reported 1004
  assertions against a solo run's 11 -- all 1000 `try_get`s returned true, so the
  scan ran out of budget on a queue that was never empty. The fix drains the queue
  to empty. Waiting for the frame, which this page recommended, would have waited
  on a frame that had already arrived.

  **LOCALAI WORKER — must be DOWN for any timing/benchmark work on this box
  (user-directed 2026-07-22).** It is a **root LaunchDaemon**, not a container
  and not a user LaunchAgent:

  | | |
  |---|---|
  | Unit | `system/com.localai.worker` |
  | Plist | `/Library/LaunchDaemons/com.localai.worker.plist` (root:wheel) |
  | Program | `/Users/mudler/local-ai/local-ai worker` (a NATS-driven worker) |
  | Properties | `keepalive | runatload` — so `kill`ing the PID is NOT enough, launchd restarts it |
  | Log | `/Users/mudler/local-ai/worker.log` |
  | State observed 2026-07-22 | **running**, PID 327, RSS ~51 MB, up 1d08h, **MEASURED idle: 0.0% CPU, and `ioreg IOAccelerator PerformanceStatistics` reports `Device Utilization % = 0`, `Renderer Utilization % = 0`, `Tiler Utilization % = 0` — it holds NO GPU work.** Log shows only periodic `NATS backend.list` events (~1 per 6 h); no model loaded |

  ```sh
  # inspect (works WITHOUT root)
  launchctl print system/com.localai.worker
  # stop  (NEEDS root; bootout, because KeepAlive would restart a killed process)
  sudo launchctl bootout system/com.localai.worker
  # restore to the observed state
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.localai.worker.plist
  launchctl print system/com.localai.worker | grep state   # expect: running
  ```

  **NOT STOPPED during W0**, for two reasons, both recorded deliberately:
  (1) stopping it needs an interactive `sudo` password and this box has no
  passwordless sudo (`sudo -n true` -> "a password is required"), so an agent
  cannot do it unattended; (2) W0 took **no timing measurement whatsoever** —
  every gate is a functional/correctness assertion — so contention could not
  affect any recorded result. **The next agent doing MLX-vs-ours benchmarking
  MUST get the user to run the bootout above first; any Metal timing taken with
  this daemon up is VOID.** Note also three `actions.runner.localai-org-*` GitHub
  Actions runners as user LaunchAgents (PIDs 599/600/601) which can start CI jobs
  on this box at any time — quiesce those too before a benchmark series
  (`launchctl bootout gui/$UID/actions.runner.localai-org-<name>.<label>`).

  **STILL NOT STOPPED as of the 2026-07-22 MLX baseline run** — same reason
  (no passwordless sudo). The MLX numbers in
  [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) are therefore recorded
  **`BLOCKED-ON-SUDO` / INDICATIVE, not binding**; the recipe is a one-command
  re-run once the user boots the daemon out. **Second contender found the same
  session and not anticipated by the earlier note:** the desktop **aerial video
  wallpaper** — `WallpaperAerialsExtension` (PID 472, **8.2% CPU**) plus
  `VTDecoderXPCService` (PID 518, 2.2%) — decodes video continuously and touches
  the GPU; it is the actual source of the ~1.47 load average on an otherwise idle
  box. Disable it (System Settings -> Wallpaper, or log the console user out)
  before any binding run. It was left untouched.

  **MLX IS NOW INSTALLED (2026-07-22), via the venv route as recommended** —
  brew was NOT used, so `python@3.14` never entered `/opt/homebrew/bin` and the
  PATH our macOS builds use is unchanged:

  ```sh
  /usr/bin/python3 -m venv ~/mlx-venv && ~/mlx-venv/bin/pip install -U pip mlx-lm
  ```

  | | |
  |---|---|
  | Resolved versions | **`mlx` 0.29.3, `mlx-metal` 0.29.3, `mlx-lm` 0.29.1** — the CLT python 3.9.6 caps the resolve BELOW brew's 0.32.0. Record this: an unpinned competitor arm is not a floor |
  | Location | `~/mlx-venv` (off every build PATH), `~/hf-cache` (3.2 GB model cache) |
  | Model | `mlx-community/Qwen3-1.7B-bf16` @ rev `9cd6692855d3e06772228e9a962b2606359b2d24` |
  | Ships prebuilt | `mlx/lib/mlx.metallib` **104,894,650 bytes** + `libmlx.dylib` — so CONSUMING MlX needs no Xcode, but BUILDING it from source does (`xcrun metal`), which this box cannot do |
  | Device probe | `mx.metal.device_info()` -> `applegpu_g16g`, `max_recommended_working_set_size` 12,713,115,648 (11.84 GiB), `max_buffer_length` 9,534,832,640 |
  | Removal | `rm -rf ~/mlx-venv ~/hf-cache` — neither is on any PATH our builds consult |

- **Local AMD/ROCm profile (gfx1200)**: Radeon RX 9060 XT, `gfx1200` (RDNA 4),
  ROCm 7.2.3, AMD Ryzen 7 5800XT host. It is not a fleet device, so `rc` does
  not manage it and GPU work takes the file mutex `${GPU_LOCK:-$HOME/gpu.lock}`
  directly. Build it from `nix develop .#rocm-shell`, which prints the
  configure and gate recipe on entry, and read the arch from `rocminfo`'s
  `Name:` field rather than from `CMakeCache.txt`. This box is the gate
  environment for the `BACKEND-ROCM` rows.

  **The `ROCm` known-red baseline. A row on this box does NOT own these.**
  Same key and same columns as the sm_110 table above: the pair is
  `(name, failure mode)`, the table is keyed on NAME, and the line number is a
  pointer to re-derive rather than part of the key.

  | Test | Mode | First failing assertion | Cause |
  |---|---|---|---|
  | `test_backend_cross_device` | Failed | `:2067` `CHECK(got == ref_b)`, case "MoeSiluMul matches the CPU oracle within NMSE <= 5e-4", bf16 arm, logged `std::string(DeviceName(dt)) := ROCM`. 26 cases / 25 passed, 80253 assertions / 1 failed | [#1954](https://github.com/mudler/vllm.cpp/issues/1954), the gfx1200 counterpart of the CUDA-only [#1802](https://github.com/mudler/vllm.cpp/issues/1802), which carries this same test name and this same assertion on sm_110. [#907](https://github.com/mudler/vllm.cpp/issues/907) is the same DEFECT FAMILY on sm_121a and NOT the same test: it records `test_cuda_ops`'s "CUDA silu_and_mul matches CPU" failing at 439 of 440 assertions, a different test and a different assertion. Elements differ in the last digit |

  #1954 recorded that assertion at `:2063`, which is where it sat before this
  row's `CAPTURE` repair added four comment lines above it. The move is the
  reason the key is the name and not the line.

  Measured 2026-08-25 on `row/ROCM-KQUANT-NWARPS-DECODE`, gate
  `ctest --test-dir build-hip -R 'rocm|cross_device'`, 5 tests / 4 passed / 1
  failed. The assertion count and the pass and fail split have not moved since
  `b06928af4`, and that control was run twice: once by a fresh review, once by
  the implementer of `d0474321c` while writing this note. Two runs, not two
  reviews. The line number and the `ROCM` rendering in the row above are read
  from `d0474321c` instead, because that is the commit that moved the assertion
  and repaired the `CAPTURE` spelling; at `b06928af4` the same failure sits at
  `:2063` and logs `DeviceName(dt) := 1`. Proven pre-existing rather than
  assumed: reverting that row's two source files to the parent `5888abf11` and
  rebuilding reproduces the identical failure, at 24 of 25 cases and 1 of 80195
  assertions. **Report this gate as one expected red, never as a clean pass**,
  and re-measure whenever the base SHA moves across `src/`, `include/` or
  `tests/`.

## Benchmark models on Ettore's dgx.casa profile

- `~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4`
  (snapshot complete, ~22G, 3 safetensors shards — re-downloaded 2026-07-03
  after the original snapshot was found incomplete)
- `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4`
- `~/work/apex/qwen36_35b/Qwen3.6-35B-A3B-APEX-*.gguf` (GGUF-gate inputs)
- `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` (fast tests)

## Gate model architecture (from GGUF metadata, arch `qwen35moe`)

40 blocks = 10 × (3 GDN + 1 full-attn); hidden 2048; full-attn GQA 16q/2kv,
partial RoPE 64 dims (MRoPE sections [11,11,10,0]), rope base 1e7; MoE 256
experts top-8 + 1 shared (expert FFN 512); GDN: conv kernel 4, 16 groups,
inner 4096, state 128; context 262144.

## Prior art on Ettore's dgx.casa profile (mudler's llama.cpp patch series — mine for GB10 kernels)

- `~/killgate_series/` — NVFP4 W4A4 FP4 MMA prefill, qwen35moe NVFP4
  quant/dedup, MoE decode regraph
- `~/llama-phase93-qwen3next-gqa-bcast`
- `~/llama-phase84-attn-only-source`

## TODO

- Binding immutable `3f256ab` remains **55/124 axes pass, 69 fail** against
  vLLM v0.25.0. Finalized c2 root `179a0fc` already maps the executed path and
  selects the complete **193 vs 97** GDN projection mismatch. W1 merged BA is
  implemented/`GATING`. Clean pushed `581d335` under
  `~/work/vllm.cpp-gdn-ba/immutable-581d335…` passes the exact CUDA 13.0.88 /
  CUTLASS / Triton-AOT build, packed F32/BF16 capture/replay, strict memcheck,
  merged/split 27B and inert native-35B gates; the isolated BF16/decomposed
  control fails the token near-tie. Immutable `0091cd1`, finalized by pushed `8a1f923`, is
  `complete-structural`: both exact-c2 arms pass all 24 local range contracts at
  merged 963/145 versus split 1,011/193, with 48 BF16-only removals and unchanged
  selected non-BF16 families. Clean `f344dec` closes packed W1D2/G2 for
  default+rollback 27B **235/235**, 35B/GGUF inertness and strict safety.
  `benchmark_binding=false`. Close paired node traces and the c2/c16 component
  before qkvz.
  Independently remove **22.920
  GiB** host-weight mirror and overlapping source pages. No 35B performance
  command runs before all 27B axes pass.
- Keep the existing SGLang v0.5.13 P1 evidence immutable. The distinct
  shared-prefix gate pins v0.5.15 `f63458b` and image digest `d0a667e`; its PX1
  deterministic 64k/256k harness/counter work is ready after the priority
  cache-off closure. Write the dedicated `KV-MAMBA-ALIGN` spike before PX2,
  then require matched BF16/no-spec capacity, native hit/no-eviction evidence,
  full axes and traces. Never mutate the vLLM oracle while provisioning SGLang.
- ~~Bootstrap CMake + MLX on the M4 host before the Metal backend bring-up.~~
  **RESOLVED/SUPERSEDED 2026-07-22** by the [backend fan-out
  spike](specs/backend-fanout-metal-vulkan-xpu.md): CMake is already present,
  and MLX is **not** a bring-up prerequisite (native MSL compiles at runtime
  with CLT only, so E2 precedes E1). The real prerequisite is spike work item
  `W0` — chiefly the `CMakeLists.txt:304-306` Apple `-force_load` fix, without
  which every static registrar is silently dropped on macOS and even the CPU
  backend fails to register. `brew install mlx` is deferred to work row `M5`.
  **CLOSED 2026-07-22: `W0` LANDED** — the `-force_load` fix is in and
  `test_backend` is 7/7 on the M4, so the M4 is fully usable for backend work.
  **REOPENED in a different role:** MLX must now be installed on the M4 as the
  **competitor BENCHMARK arm** (user directive; `BACKEND-GATE-METAL-MLXLM`),
  which is independent of its demotion as an implementation path. Use the venv
  route recorded in the M4 entry above, and stop the LocalAI worker daemon
  first.
- **Vulkan runtime is already usable and needs no acquisition.** dgx GB10
  enumerates as a real Vulkan `INTEGRATED_GPU` at API 1.4.312 (loader 1.4.328 +
  NVIDIA ICD) with `VK_KHR_cooperative_matrix` v2 and `VK_NV_cooperative_matrix2`;
  the dev box enumerates `llvmpipe` (Vulkan 1.4.318, CPU) for GPU-free CI.
  Optional still: `libvulkan-dev` and `vulkan-tools` (neither is needed to build
  or gate — the backend `dlopen`s the loader and vendors the Khronos TYPE headers).
- **Vulkan shader toolchain — glslang 16.5.0, installed 2026-08-06 (`VK-A1`).**
  `$HOME/tools/glslang-16.5.0/bin/glslang`, from the upstream prebuilt Linux
  x86_64 release tarball; no root, nothing linked (it is a build-time tool, never
  a dependency — `.agents/discipline.md`). Put that directory on `PATH` to
  regenerate committed SPIR-V with `scripts/gen-vulkan-spirv.py`.
  **Two measured facts about the pin.** (1) `src/vt/vulkan/vulkan_spirv.h:16`
  records `Glslang Version: 11:16.4.0`, but **16.4.0 ships NO release assets** —
  only `16.5.0` and `main-tot` do, and Ubuntu packages `15.1.0` — so the recorded
  version cannot be fetched and cannot back a CI gate. (2) The committed SPIR-V
  nonetheless reproduces **byte-for-byte under 16.5.0** (`--check` passes, exit 0),
  which proves the committed artifact is what it claims AND that the emitted
  SPIR-V is stable across a glslang minor bump. The freshness gate therefore pins
  the DOWNLOAD URL rather than asserting a version string.
  The older note here — that Ubuntu's shaderc 2023.8 `glslc` is too old for the
  coopmat2 feature probe — still holds and is why the system package is not used.
- **dgx Vulkan/llama.cpp comparison toolchain (2026-08-07, `VK-E`).** apt:
  `glslc`, `glslang-tools`, `libvulkan-dev`, `spirv-headers`. **Ubuntu's `glslc`
  is shaderc 2023.8 and is NOT USABLE for a fair llama.cpp-Vulkan build** — it
  disables FOUR of five fast paths (`GL_NV_cooperative_matrix2`,
  `..._decode_vector`, `GL_EXT_integer_dot_product`, `GL_EXT_bfloat16`), leaving
  only `GL_KHR_cooperative_matrix`, and the build still configures and runs. A
  source-built shaderc `v2026.4-dev` lives at `/tmp/shaderc/b/glslc/glslc`; pass
  `-DVulkan_GLSLC_EXECUTABLE=` to it. **Verify the runtime banner says
  `matrix cores: NV_coopmat2` before trusting any number.** llama.cpp is
  unpacked at `~/lcpp-vk` with `build-vk/bin/llama-bench` built, **and that tree
  is the SUPERSEDED fork `237ad9b96`, not the pin.** Do not reuse it. The
  llama.cpp oracle is stock `b10451` since 2026-08-16
  ([`oracles/llama-cpp.md`](oracles/llama-cpp.md)). `237ad9b96` is a local-only
  commit on the developer's `localai-paged` branch, 65 of our own performance
  commits past upstream `b9827`, built from a working tree with 27 uncommitted
  entries. `~/lcpp-vk` therefore reproduces neither the pin nor any identifiable
  object. The `BENCH-VK-LLAMA` decode `4.36 vs 4.35 MET` measured with it is the
  **most fragile verdict** in the enumeration, a 0.23% margin inside a 0.69%
  spread, and re-taking it is owed under
  [#1003](https://github.com/mudler/vllm.cpp/issues/1003). Unpack the pinned SHA
  fresh and assert `git status --porcelain` empty before recording any number.
  Enumeration and the clean-tree rule:
  [`specs/oracle-llamacpp-repin-stock.md`](specs/oracle-llamacpp-repin-stock.md).

- **No Intel GPU exists on any box here**, so `BACKEND-XPU` end-to-end work is
  HW-BLOCKED; only policy-port, compile coverage and oneAPI CPU-device unit
  numerics are available.

## 2026-10-07: run test_kolibri1_w3 with a thread cap on shared hosts

A fresh mutation-review run of `test_kolibri1_w3` executed UNCONSTRAINED
(128 threads) OOM-killed a 256 GB host: RSS exceeded 76 GB, `dmesg` recorded
the oom-kill, and the test died with exit 137. A capped run
(`VLLM_CPP_CPU_THREADS=8`) reached the same ~76 GB anon RSS
(`total-vm:153529600kB, anon-rss:75975936kB`), so the resident floor is the
model, not the thread count: W3 needs a host with ~80 GB free above whatever
else is resident. Run it with a thread cap, and only when `free -g` reports
that much available. The constrained run is also the one whose expected
fingerprints the gates quote.
