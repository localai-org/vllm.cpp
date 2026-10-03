# LTX25-DEVICE-RESIDENCY — staging the ranked levers into work that lands, and why the instrument is stage zero

Row: `LTX25-DEVICE-RESIDENCY`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)).
Issue: [#1264](https://github.com/mudler/vllm.cpp/issues/1264).
Base: `27d5432f9`.

**This row exists because [`ltx25-decode-speed.md`](ltx25-decode-speed.md) says,
in its own stop conditions, that it has none of what is needed to act:** *"Do not
implement any lever from this row. It has no implementation authority and no
fresh review."* That spike ranked the levers, read every oracle, recorded
provenance per number and filed thirteen issues it owns. It is cited throughout
and **deliberately not restated**. Where a fact belongs to it, this row links
rather than re-derives; where this row disagrees with it, §*Our baseline* says so
with a `file:line` at this base.

## Now

`ACTIVE`. **W0 has landed and W1 is still next**, which is a lease on a box with
the checkpoints — `dgx.casa`, per `H3`. Each stage is dispatched to its own fresh
implementer and its own fresh reviewer.

**W5 landed out of order, and the order below is NOT amended by it.** It was
dispatched while a render held `dgx:gpu0` and with the lease explicitly withheld,
so W1's precondition was never met. The reconciliation is in §W5.0 and it is a
scope limit rather than a waiver: W1 gates the RANKING, and W5 makes no ranking,
ratio or speed claim at all — it lands the primitive whose absence is why #1007
was filed, a `vt::Conv3d` op with a CPU and a CUDA arm, plus the dispatch that
reaches it from the render path. The only quantities it measured are a byte
comparison and a dispatch count. **W1 still decides where the decode ranks**, and
if its table puts `decode.video` low then W5 is correct, reached, and simply not
the lever. W2, W3 and W4 are unstarted.

**What W0 answered, and what it did not.** A completed render now writes
`<output_dir>/phase-log.json` on the shipped default: a flat, non-overlapping
timeline of named phases, each carrying a monotone timestamp, a duration, a peak
host byte count and a peak device byte count, with `unaccounted_seconds` emitted
beside the sum rather than smeared over the phases. On a completed 64x64/9-frame
render through the `vllm.h` video ABI the named leaves account for **99.94% of
wall** (4.459892 s of 4.462572 s, residue 2.68 ms over 33 entries), so **the
phases sum and W1 may start**. That is the artifact's own run and the figure
`## Outcome — W0` records. **The RATIO is what the gate reads. No wall here is a
benchmark** — the same binary at the same geometry has measured 0.147 s, 0.158 s,
1.676 s, 4.463 s, 6.138 s and 12.030 s on a contended box, and the first two of
those are one minute apart. The artifact is
[`benchmarks/demo/ltx25_phase_log_fixture_cpu.json`](../../benchmarks/demo/ltx25_phase_log_fixture_cpu.json)
and its provenance is in `## Outcome — W0` below.

**What W0 did NOT measure is the device column, and a lease will not close it.**
Every `peak_device_bytes` in the artifact reads the `-1` "no probe answered"
sentinel. Two reasons are stacked. The render W0 could take was the CPU arm,
where `-1` is the correct answer. The one that matters is that **`CudaBackend`
does not override `vt::Backend::DeviceMemoryInfo` at all** —
[#1126](https://github.com/mudler/vllm.cpp/issues/1126), and `grep -rn
'DeviceMemoryInfo' src/vt include/vt` returns the base declaration
(`include/vt/backend.h:94`) and exactly one override, ROCm's
(`src/vt/rocm/rocm_backend.hip:373`). A CUDA render on `dgx:gpu0` would print
`-1` in that column too.

W0 did not wire it, and the reason is written down where the seam is:
`include/vllm/platforms/interface.h:68-72` records that CUDA's absence from that
seam is load-bearing, because `Gemma4MoE`'s device-expert LRU is its only
consumer. **Which arm of that LRU the override would wake is narrower than this
row first claimed, and the narrower statement is the one to carry.** The bf16
arm, `EnsureGemma4Fp8ExpertOnDevice` (`src/vllm/model_executor/models/gemma4_moe.cpp:548`),
is dead on CUDA for a SECOND and independent reason: it refuses at `:571` on
`vt::HasMatmulBTAlphaBeta`, whose only implementation in this tree is ROCm's
([#1205](https://github.com/mudler/vllm.cpp/issues/1205)), so wiring
`DeviceMemoryInfo` does not wake it at all. What the override WOULD wake is the
**FP8-native arm**, `EnsureGemma4Fp8NativeOnDevice` (`:611-628`), which reaches
`DevExpertLru::MakeRoom` behind no device gate other than the probe itself and
whose budget defaults to a 2048 MiB fill-only device cache when
`VT_GEMMA4_EXPERT_VRAM_MB` is unset (`:416-432`). So the override is still a
behaviour change with its own measurement, on one named arm rather than on
"every CUDA model". That is #1126's change, not an instrument's. **Until it lands, W1 samples
`--query-compute-apps=used_memory` per phase beside the table** — which is the
fallback this spec's W6 row already names, and the only device-memory instrument
GB10 answers. `## Owed` states what the column cannot report and why.

The stage order is **W0 (landed) → W1 → W2 → W3 → W4 → W5 → W6 → W7**, with the
oracle lane **O1** running beside it and blocking only ratios. **W1 is a gate on the
order itself.** The ranking every later stage inherits was measured on a host
that stopped answering ([#1040](https://github.com/mudler/vllm.cpp/issues/1040)),
and two of the levers it ranked have since landed without re-measurement, so the
profile the ranking describes is not the profile in this tree. W1 either
reproduces the order or replaces it, and **this section is amended with the
re-derived order before W2 starts, unless W1 reports `BLOCKED`, in which case W2
and W3 proceed on their byte-compare and RSS gates and W4/W5 do not start**. A
lever whose rank does not survive W1 is re-placed or dropped here, in writing,
rather than carried.

## Scope

**In scope.** Implementation ownership of the device-residency levers of an
LTX-2.5 render on GB10: the phase instrument, the retrievable-evidence contract,
the load-time memory defects, the DiT staging path, a device arm for the text
conditioning stage, a device arm for the video VAE decode, and the attribution of
the ~59 GiB. Fourteen issues, staged in §*Work breakdown*: the thirteen the
spike owns ([#1007](https://github.com/mudler/vllm.cpp/issues/1007),
[#1009](https://github.com/mudler/vllm.cpp/issues/1009),
[#1010](https://github.com/mudler/vllm.cpp/issues/1010),
[#1011](https://github.com/mudler/vllm.cpp/issues/1011),
[#1012](https://github.com/mudler/vllm.cpp/issues/1012),
[#1014](https://github.com/mudler/vllm.cpp/issues/1014),
[#1015](https://github.com/mudler/vllm.cpp/issues/1015),
[#1016](https://github.com/mudler/vllm.cpp/issues/1016),
[#1021](https://github.com/mudler/vllm.cpp/issues/1021),
[#1024](https://github.com/mudler/vllm.cpp/issues/1024),
[#1040](https://github.com/mudler/vllm.cpp/issues/1040),
[#1202](https://github.com/mudler/vllm.cpp/issues/1202),
[#1210](https://github.com/mudler/vllm.cpp/issues/1210)) plus
[#1269](https://github.com/mudler/vllm.cpp/issues/1269), which this row filed
because the defect had an owner in prose and no issue anywhere (§*Our baseline*
finding 2).

**Out of scope, deliberately.**

* **Any engine code in this pull request.** This change is the spec, the issue
  index rows and the one matrix cell the campaign invalidates. Each stage below
  opens its own `row/LTX25-DEVICE-RESIDENCY-W<n>` branch.
* **[#655](https://github.com/mudler/vllm.cpp/issues/655)** — registering
  Lightricks `ltx_core` as an admitted oracle. It is a policy row that predates
  this campaign, it is not this campaign's to close, and O1 records what this
  campaign needs from it rather than taking it.
* **Any parity ratio against any oracle.** The spike §7 establishes that no
  admitted oracle is installed on the gate host and that a same-tool both-sides
  LTX-2.5 profile is not possible today. Every stage below measures our engine
  against our engine, which is what a defect against our own structure needs, and
  says so.
* **[#1164](https://github.com/mudler/vllm.cpp/issues/1164)**, denoise-loop
  graph capture. W7 is its *decision point*, not its implementation.
* **The diffusion VAE decode arm** (`NADiffusionDecoder`). `Ltx2VideoDecode`
  refuses `kDiffusion` by name today
  (`include/vllm/model_executor/models/ltx2_video_vae.h:236-244`) and that
  refusal is correct; W5 gives the **conv** decoder a device arm and leaves the
  refusal standing.

## Upstream chain

**The oracle question is settled and this row does not re-open it.** The spike
§2 read Lightricks `ltx_core` (the reference implementation), `diffusers` (which
implements LTX-2.5, both decode arms), SGLang (2.0 and 2.3, not 2.5), vLLM-Omni
(2.3, and its 2.5 route disqualified as a reference-degraded configuration) and
vLLM (implements nothing here, so AGENTS.md's *"implements nothing"* branch
applies). §7 records the denominator verdict. **None of that is re-derived here.**

What this row adds is the mapping from *stage* to *what upstream binds it*, and
the honest statement that **half the stages have no upstream to mirror at all**:

| Stage | Upstream that binds it | Where it was read |
|---|---|---|
| W0 instrument | none — no upstream emits our phase names | local seam |
| W1 re-measure | none — a measurement of our engine | local |
| W2 load memory | none for #1015/#1016; upstream offloads transformer weights and never the VAE (`installation.md:90`) | spike §6 lever 5 |
| W3 staging | none — `Alloc`/`Copy`/`Synchronize` is our seam | local |
| W4 text tower device arm | `feature_extractor.py:85-129`, `encoder_configurator.py:187,206-208` for the extractor contract this port already mirrors | already ported; see `include/vllm/model_executor/models/ltx2_text_encoder.h:282-293` |
| W5 VAE decode device arm | `blocks.py:1139` + `single_gpu_model_builder.py:273`; `decoding_av.py:71`; `interface.py:92`; `conv_video_decoder.py:282-286` (DTYPE only — see §W5.10); `normalization.py:32-40` | spike §6 levers 1 and 2, which read them |
| W5 rider (#1011) | `memory_efficient_decode.py:1-20`, `:91-105`, `:122-204`, `:234-248`, `:541-609`, `:617-627`; default `True` at `blocks.py:1059` | spike §6 lever 4, which read them |
| W6 residency | upstream offloads the transformer, never the VAE (`installation.md:90`) | spike §6 lever 5 |
| O1 oracle record | AGENTS.md oracle table; [`.agents/oracles/diffusers.md`](../oracles/diffusers.md) | — |

**Every upstream anchor in that table is cited from the spike, which read those
files. This row read none of them.** Naming a line I did not open would be the
same defect as quoting a number I did not measure. The stage that takes each row
re-derives its own anchors against the pinned oracle before it writes code, per
[`.agents/porting.md`](../porting.md), because line anchors go stale and these
are already one campaign old.

**Where there is no upstream, the stage says so and gates against our own prior
arm.** That is the same footing `LTX25-DECODE-THREADS`
([`ltx25-decode-threads.md`](ltx25-decode-threads.md) §1) stood on, and it is
legitimate precisely because a single-threaded host stage on a `--device cuda`
render is a defect against *our* structure, not a divergence from upstream's.

## Our baseline

The state of the tree at `27d5432f9`, re-derived rather than inherited. Four of
these are corrections to the framing this campaign was handed, and each names the
line that settles it.

**1. The denoise IS device-routed, so #1024 is a symptom with more than one
candidate producer, and the code has already eliminated one of them.**
[#1024](https://github.com/mudler/vllm.cpp/issues/1024) is titled *"stages 35.5
GiB onto the GPU and then never uses it"*, which reads as an unrouted forward.
It is not: `src/vllm/multimodal/ltx2_video.cpp:847` selects
`Ltx2StreamDitToDevice` on `im.on_device`, and `:3864-3866` selects
`Ltx2DitForwardDevice` over `Ltx2DitForward` on the same flag.
`Ltx2DitForwardDevice` (`src/vllm/model_executor/models/ltx2_device.cpp:1111`)
runs a real device forward — it takes `vt::Queue&`
(declared `include/vllm/model_executor/models/ltx2_device.h:136`) and obtains the
backend inside it via `vt::GetBackend(queue.device.type)` at
`ltx2_device.cpp:1146`, calls `CheckWeightsResident(weights, queue.device)` at
`:1144`, and drives 48 blocks of device ops. The spike's own §5 records that the
sampled window **had written no frame**, which excludes the VAE decode as well.
So the 1.00-core, GPU-zero window is *neither* the routed denoise *nor* the
decode, and the remaining candidates
are the conditioning stage (finding 2), the LoRA fuse
([#1202](https://github.com/mudler/vllm.cpp/issues/1202),
[#1210](https://github.com/mudler/vllm.cpp/issues/1210)) and the load path
itself. **Nothing in the tree picks between them, and that is exactly what
[#1010](https://github.com/mudler/vllm.cpp/issues/1010) is.** This is the single
strongest argument for putting the instrument first: the campaign's headline
measurement does not identify its own subject.

**2. The conditioning stage is host-only by TYPE, and it was in nobody's lever
table.** `src/vllm/multimodal/ltx2_video.cpp:2085`, `:2799` and `:4638` each
build `vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}`;
`src/vllm/model_executor/models/ltx2_text_encoder.cpp:446` builds a **second**
hard-coded CPU queue inside the extractor, so changing the call sites alone would
not move the projection. The comment at `:443-445` states the reason and it is
the real blocker: the tower's weights are `Gemma4Weights`
(`include/vllm/model_executor/models/ltx2_text_encoder.h:542-548`) of
`OwnedTensor` (`include/vllm/model_executor/models/gemma4.h:114-125`) over
`OwnedBytes` (held at `include/vllm/model_executor/models/qwen3_5_weights.h:47-56`,
defined at `include/vllm/model_executor/models/owned_bytes.h:42`), which has no
device field — `grep -icE 'device|cuda|vt::Queue' owned_bytes.h` returns **0**.
`grep -c 'Device\|kCUDA\|Queue'
src/vllm/model_executor/models/gemma4_weights.cpp` returns **0** against a
positive control of **16** for the identical pattern in `ltx2_device.cpp` and
**4** in `ltx2_text_encoder.cpp` itself, so the zero is a finding and not a
mistyped pattern. `LTX25-TEXT-LINEAR-SEAM`
([`ltx25-text-linear-seam.md`](ltx25-text-linear-seam.md) `## Owed`) found this
and wrote *"it needs its own issue"*; none existed, so this row filed
[#1269](https://github.com/mudler/vllm.cpp/issues/1269) and stages it as W4. The
spike's §6 lever table does not carry it at all — it post-dates the table.

**3. [#1015](https://github.com/mudler/vllm.cpp/issues/1015) does NOT gate device
residency; [#1016](https://github.com/mudler/vllm.cpp/issues/1016) does.** #1264
groups them as *"the two memory defects that must land before device residency
has headroom"*. The device path **refuses the widen by name**:
`src/vllm/model_executor/models/ltx2_loader.cpp:759-765` fails
`Ltx2StreamDitToDevice` on `widen_to_f32`, and `:780-800` keeps exactly one host
buffer live per tensor. So #1015's double-hold — `Ltx2WidenDitToF32` at
`:739-754` pushes the f32 copy onto `checkpoint.storage` at `:752` while the bf16
buffer pushed at `:729` is never dropped — is a **host-arm** defect. It is still
in W2 and still early, because the host f32 arm is the *reference* every W4 and
W5 correctness gate compares against, and an arm that cannot load is not a
reference. #1016 is the one that gates device headroom: on GB10's unified memory
the device copy and the dead file cache come out of the same 119 GiB, and the
LTX loaders call `MaybeReleaseSourcePages` nowhere while 14 other files under
`src/vllm` do — `grep -rl` returns 15 files, one of which is the defining
`safetensors_reader.cpp` and one of the remaining 14 a header
(`safetensors_reader.cpp:339` is the definition;
`qwen3_5_dense_weights.cpp:88` and `gemma4_weights.cpp:149-150` are two of the
callers). Note the fix is armed by default —
`LoadWindowedReleaseEnabled()` (`safetensors_reader.cpp:303-312`) reads
`VT_LOAD_WINDOWED_RELEASE` and defaults ON — so adding the call changes behaviour
on the shipped default, which is why W2 gates peak RSS on both arms rather than
assuming an improvement.

**4. Two of the ranked levers have LANDED since the ranking, and neither was
re-measured, so the ranking is stale by construction and not merely
unverified.** [#1009](https://github.com/mudler/vllm.cpp/issues/1009) (lever 3)
closed via [#1041](https://github.com/mudler/vllm.cpp/issues/1041): three
`vt::cpu::ParallelForRows` dispatch sites are live in
`src/vllm/model_executor/models/ltx2_video_vae.cpp:170`, `:218` and `:276`.
[#1208](https://github.com/mudler/vllm.cpp/issues/1208) closed via
`LTX25-TEXT-LINEAR-SEAM`, which measured one conditioning pass from **671.777 s
of one core to 78.421 s (8.57x)** on x86 and records the projection as
**39-100%** of the ~1731 s resolution-constant phase
([#1087](https://github.com/mudler/vllm.cpp/issues/1087)) — a bound, because the
GB10-to-x86 per-core ratio was never measured. `docs/BENCHMARKS.md:494` carries
that bound. (**Dead by RESTRUCTURING, re-resolved 2026-09-01 by row
`LTX25-ANCHOR-REPAIR`**: `docs/BENCHMARKS.md` is now an 18-line index over
`docs/benchmarks/<benchmark-id>.md`, so `:494` has no successor line. `:494` sat
under that file's `## Open gaps` heading -- recorded independently in
`.agents/specs/public-docs-readability.md` -- which is today
[`docs/benchmarks/open-gaps.md`](../../docs/benchmarks/open-gaps.md). The bound
is cited there by section, not by coordinate.) **Neither landing has been re-measured end to end**, and
`LTX25-DECODE-DTYPE`'s speed magnitude was never measured either
([`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) `## Owed`). The spike's own
re-ordering note (*"the remaining order is 6, 3, then 1"*) predates #1208
entirely.

**5. The video VAE decode arm does not exist, and the signature says so.**
`Ltx2VideoDecode` and `Ltx2ConvVideoDecode`
(`include/vllm/model_executor/models/ltx2_video_vae.h:230-244`) take
`const std::vector<float>& latent` and no queue; `grep -c 'Queue'
src/vllm/model_executor/models/ltx2_video_vae.cpp` returns **0** against a
positive control of **6** `vt::` hits in the same file and 2 `Queue` hits each in
the sibling `ltx2_dit.cpp` and `ltx2_device.cpp`. The spike's reading stands: the
arm does not exist, so W5 is a new op family and not wiring.

**6. The positive control [#1024](https://github.com/mudler/vllm.cpp/issues/1024)
owes is, as far as this tree records, still owed.** The spike states plainly that
it has no control proving `utilization.gpu` reads high for a real kernel on GB10,
and that `--query-gpu=memory.used` returns `[N/A]` on that box (§4.3).
`grep -rn '11\.25\|power\.draw\|board power' .agents/ docs/` returns only
unrelated matches, so no such control is written down anywhere. **W1 takes it and
records it**, using the two instruments that do work on GB10 —
`--query-gpu=power.draw,temperature.gpu` and
`--query-compute-apps=used_memory` — beside the host-side thread states and
`utime`/`stime` rates, so the two sides corroborate rather than one carrying the
claim alone.

## Port map

Anchors at `27d5432f9`. Each stage re-derives them on its own branch before it
edits, because several already drifted between the spike's base and this one
(the spike's `ltx2_video.cpp:2946`/`:2948` is now `:3864-3866`; its
`ltx2_loader.cpp:694-710` is now `:739-754`; the seam row's `ltx2_video.cpp:4479`
is now `:4638`).

**That re-derivation is the ONLY check these anchors get, and the gap is
specific.** No checker in this tree reads a bare `path.cpp:123` written in prose.
`check_links` (`scripts/check-agent-record.py:862-886`) parses markdown link
targets only, and validates a `#L<n>` fragment against the target's line count —
so a dangling link and an out-of-range `#L999999` are both caught, and a bare
`ltx2_video.cpp:99999` in a sentence is not read at all. `check_spec`
(`scripts/check-agent-record.py:1169`) takes a `ClaimRow`, so its
structured-section requirement reaches only a spec linked from a matrix row, and
this row has none (§*Decisions taken here*, "No matrix row, and no claim file").
Every `file:line` below and in §*Our baseline* is therefore checked by a person
reading it and by nothing else — four of them were off by one or two lines, in
both directions, when this spec was first reviewed. Each stage re-derives with
`grep -n` against its own base and pastes the output into its report, rather
than counting by eye.

| Stage | Our anchor at this base | What changes | Production entry point it must be reached from |
|---|---|---|---|
| W0 | `src/vllm/multimodal/ltx2_video.cpp` render driver; `include/vllm.h` | phase-boundary timestamps + peak host/device bytes emitted per phase | `examples/ltx2_gen` through the `vllm.h` video ABI, on the shipped default |
| W1 | none — measurement only | none | the same binary W0 shipped |
| W2a #1016 | `src/vllm/model_executor/models/ltx2_loader.cpp:722-735`, `:780-800` | `MaybeReleaseSourcePages` after each materialization, both arms | `Ltx2LoadDitFromSafetensors` / `Ltx2StreamDitToDevice` via `ltx2_video.cpp:847` |
| W2b #1015 | `src/vllm/model_executor/models/ltx2_loader.cpp:739-754` | drop the bf16 source buffer as each view is repointed | the host f32 arm at `ltx2_video.cpp:847` |
| W2c #1210 | `src/vllm/model_executor/models/ltx2_loader.cpp:808-912` (`Ltx2RebindDitLoras`) | delete the load-time fuse that phase 0 provably undoes | `Ltx2RebindDitLoras` on a LoRA render |
| W3 #1021 | `src/vllm/model_executor/models/ltx2_loader.cpp:780-800`; `src/vt/cuda/cuda_backend.cu:77-95` | batch/overlap the per-tensor `Alloc`+`Copy`+`Synchronize` | `Ltx2StreamDitToDevice` |
| W4 #1269 | `src/vllm/multimodal/ltx2_video.cpp:2085`, `:2799`, `:4638`; `src/vllm/model_executor/models/ltx2_text_encoder.cpp:446`; `include/vllm/model_executor/models/gemma4.h:114-125` | a device weight arm for the tower + queue threading | `Ltx2EncodePromptToConditioning` from the render driver |
| W5 #1007 | `src/vllm/model_executor/models/ltx2_video_vae.cpp:161-290`; `include/vllm/model_executor/models/ltx2_video_vae.h:230-244`; `include/vllm/model_executor/models/ltx2_tiling.h:308` | a `vt::` conv3d op with CUDA and CPU arms; a device `Ltx2VaeWeights` | `Ltx2VideoDecodeStreaming` from the render driver |
| W5 rider #1011 | same, plus the NDHWC half owed to [`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) `## Owed` | workspace reuse, in-place norm/SiLU, free-before-conv, temporal chunking, memory format | same |
| W6 #1014 | W0's peak-memory emission; `--query-compute-apps=used_memory` | attribution, then a release if one is found | the render |
| W7 | `src/vt/cuda/cuda_backend.cu:206-222` (`BeginCapture`, `EndCapture`, `Replay` — the capture primitive) | nothing unless the measurement says capture | — |
| O1 #1012 | [`.agents/oracles/diffusers.md`](../oracles/diffusers.md) | record that `diffusers` implements LTX-2.5 and measure its gateability | — |

**Nothing lands dead.** Each stage's smallest failing test enters through the
entry point in the last column, at that stage's own merge commit, and each
stage's fresh reviewer deletes that production call site in a scratch copy and
reruns the focused gate — a gate that stays green without the call site measured
a class, not a capability
([`.agents/reachability.md`](../reachability.md)). W1, W6 and W7 land no code
and therefore have nothing to reach; they land evidence and a record.

## Tests to port

**There is no upstream test to port for W0, W1, W2, W3, W6 or W7.** Saying so is
the honest answer: none of them mirrors an upstream behaviour. Their tests are
local red-first cases, and each names the guarantee it pins:

| Stage | The smallest failing test, and what makes it red first |
|---|---|
| W0 | a render at a small fixed geometry through the video ABI emits a phase table whose entries carry a monotone timestamp and a byte count; red before, because the path emits one line per render ([#1010](https://github.com/mudler/vllm.cpp/issues/1010)) |
| W2a | peak `Rss_File` across a load, asserted below the pre-change value by a stated margin; red before, because no release call exists |
| W2b | peak RSS across a **host** load, asserted below the pre-change value; red before, because both buffers are retained (`ltx2_loader.cpp:729` and `:752`) |
| W2c | a LoRA load followed by a phase-0 rebind performs the fuse **once**; red before, because it fuses, un-fuses and re-fuses |
| W3 | staged bytes byte-compare equal to the serial path AND the round-trip count falls; the byte-compare is the correctness half and must be written first |
| W4 | conditioning tensors from the device arm against the host arm within a stated tolerance, entering through `Ltx2EncodePromptToConditioning`; red before, because there is no device arm to select |
| W5 | decoded **pixels** from the device arm against the current host f32 arm at a fixed seed and geometry, entering through `Ltx2VideoDecodeStreaming`; red before, because `kind`-to-device does not resolve |

**W5 and W2b do have an upstream reference and must use it.** The conv decoder's
goldens are generated by `scripts/gen-ltx2-vae-goldens.py`, and the spike §8
records the trap: that generator casts every upstream parameter to f32 at
`:223`, so the oracle itself runs f32 and a width comparison against it is
**vacuous by construction** (`ltx2_video_vae.cpp:41-44`). W5 therefore carries a
separable-reduction width case of its own, in the shape `LTX25-DECODE-DTYPE`
shipped — a `[+1e8, 0.1 x 25, -1e8]` tap set where an f32 accumulator lands on
exactly 0 in any summation order and an f64 one does not — and **must not let its
expectation coincide with the zero value of an absent computation**, because a
reachability mutation that replaced the decode with a zero-filled buffer *passed*
for that row until the expectation was biased away from zero.

Three rules bind every mutation in every stage, each earned by a failure this
campaign already had:

1. Report `git diff --stat` with the mutation, because **a mutation that never
   applied reads as a passing test**.
2. Report the compile status, because **a mutation that fails to build reads as a
   passing test** and a stale binary prints a plausible verdict.
3. Report the exit code and a non-zero case count, because `assertions: 0` is a
   skip wearing a pass and `doctest -tc` splits its filter on commas.

## Gates

**Correctness before speed, at every stage, without exception.** A decode or a
conditioning pass that is faster and different is not a win.

**The completion gate is pixels, never an exit code.**
[#1149](https://github.com/mudler/vllm.cpp/issues/1149) is the precedent: a
completed render exited 127 from a missing `ffmpeg` and read as a failure. The
shipped verification recipe is `LTX25-RESOLUTION-ENVELOPE`'s, and every stage
that completes a render reuses it verbatim: distinct per-frame md5s for every
frame, zero near-uniform and zero near-black frames, adjacent-frame mean absolute
difference against a uniform-noise reference on the same shape, zero zero-motion
pairs, and the audio track's rate, duration, dBFS and above-threshold window
count.

Per stage, with the refutation shape stated **before** the run, because AGENTS.md
forbids declaring a ceiling and a refuted lever owes the next traceable
hypothesis:

| Stage | PASS | REFUTED, and the next hypothesis |
|---|---|---|
| **W0** | one completed render emits a phase table whose durations **sum to wall within a stated tolerance**, plus per-phase peak host bytes and device bytes; the file is retrievable from this repo's evidence path | the phases do not sum. Then the instrument is incomplete and **W1 cannot start**: the missing time is a phase nobody named, and W0 iterates until it is named. A partial phase table is worse than none, because it invites the same interpolation the sampler CSVs already invited |
| **W1** | a per-phase table at **two geometries** on an idle leased box, with board power and `--query-compute-apps=used_memory` sampled beside it, that either reproduces the spike's order or replaces it; `## Now` is amended with the result; #1040's artifacts land in the repo | the phase table contradicts the ranking. That is a **success of this stage**, not a failure: the order in `## Now` is rewritten, and any stage whose lever is now small is dropped from the campaign here in writing |
| **W2a** | peak `Rss_File` across a load falls, measured on the same binary A/B, both arms; the DiT byte-compares identical after the release | RSS does not fall. Then the pages were not resident to begin with and the +10.83 GiB was a different allocation; next hypothesis is the dequantize scratch inside `MaterializeDitTensor` |
| **W2b** | peak RSS across a host f32 load falls by approximately the bf16 half of the contract; the f32 views still byte-compare identical | RSS does not fall. Then the bf16 buffers were already being freed by some path this reading missed, and the ~37.9 GiB figure is wrong; next hypothesis is `checkpoint.storage`'s own lifetime |
| **W2c** | the fuse count on a LoRA load-then-phase-0 sequence drops from two to one; fused weights byte-compare identical to the two-pass result | the weights differ. Then the load-time pass is **not** provably wasted and #1210's premise is wrong; the row stops and reports `NEEDS_DECISION` |
| **W3** | staged bytes byte-compare identical to the serial path, **then** staging wall and MiB/s improve on an idle box | wall does not move. Then the cost is not the round trips; next hypothesis is the host-side dequantize inside `MaterializeDitTensor`, which W3 must then time separately from the transfer |
| **W4** | conditioning tensors match the host arm within a stated tolerance, **then** the conditioning-pass wall falls at the shipped geometry, measured against the same binary's host arm | the conditioning pass is a small fraction of W1's phase table on GB10. Then #1269 is refuted **as a lever** while remaining a correct capability, W4 drops to the tail, and the next hypothesis is whichever phase W1's table actually shows |
| **W5** | pixels against the current host f32 arm at fixed seed and geometry, `max\|diff\|` recorded and inside a stated tolerance argued at the site; **then** decode wall; **then** a completed render verified on pixels | pixels differ beyond tolerance → the arm is wrong and does not land. Wall does not move with pixels correct → the decode was not the term W1's table said it was, and the next hypothesis is the memory format (#1011's NDHWC half), measured before any further kernel work |
| **W6** | the ~59 GiB either reproduces under W0's per-phase byte counters and is attributed to a named allocation, or it **does not reproduce**, which closes #1014 as not-reproducible-with-this-instrument | neither: it reproduces and W0's counters do not see it. Then the allocation is invisible to the counters W0 emits, which is the spike §4.3 device-class hypothesis, and the next instrument is `--query-compute-apps=used_memory` sampled per phase |
| **W7** | GPU-busy against wall on a device-resident loop. Host-dispatch-bound → capture #1164. GPU-bound → **#1164 closes as a refutation**, the way [#1161](https://github.com/mudler/vllm.cpp/issues/1161) closed prefill capture | the loop is neither, because a host phase still dominates. Then W7 does not run and the campaign returns to W1's table |
| **O1** | `diffusers` recorded as covering LTX-2.5 in its oracle file with a gateability measurement on the gate host | it does not build or run 2.5 there → `gateable = no` with the issue that owes the measurement named, per AGENTS.md |

**No stage may quote a ratio against any oracle** until O1 lands a gateability
measurement. Every number above is our engine against our engine on the same
binary, and each stage says so in its own `## Outcome`.

**The full gate before every push**, per stage, and the operator reruns it
itself: an implementer or reviewer report is an input, never a gate result. Known
reds excluded by name and by issue: `test_cpu_x86_llamacpp_floor`
([#618](https://github.com/mudler/vllm.cpp/issues/618)) and the `windows-msvc-*`
lanes ([#584](https://github.com/mudler/vllm.cpp/issues/584)).

## Dependencies

```
W0  instrument ─────────────► W1 re-derive the ranking ──┬──► W2 ──► W3 ──► W4 ──► W5 ──► W6 ──► W7 (#1164)
                                                          │
O1  oracle record (#1012) ────────────────────────────────┘  (blocks RATIOS only, never a defect fix)
```

* **W1 depends on W0** and on nothing else. Without a phase log the re-derivation
  would be another sampler interpolation, which is the exact evidence class
  [#1040](https://github.com/mudler/vllm.cpp/issues/1040) files against.
* **W2, W3, W4, W5 depend on W1** for their *order*, not for their *correctness*.
  Each is a defect against our own structure and each could be implemented today;
  what W1 decides is which is worth doing first and whether any is worth dropping.
  A stage may start early only if W1's table has already named its phase.
* **W3 depends on W2a.** Batching the staging raises transient device residency,
  and doing that before the dead file cache is released is how a 119 GiB box
  OOM-reboots.
* **W4 and W5 depend on W2b**, because the host f32 arm is the reference their
  correctness gates compare against, and #1015 makes that arm's peak
  approximately 105.9 GiB on a 119 GiB box.
* **W5 depends on W3** only for cost: without it, every W5 iteration pays 7.5
  minutes of staging before it reaches the decode. That is a schedule dependency
  and it is stated as one.
* **W6 depends on W0** for its instrument and may dissolve without any
  implementation at all.
* **W7 depends on W5** — a graph collapses host launch dispatch, and this render
  has almost none to collapse until the loop is device-resident. Its decision
  point is a measurement, not an implementation.
* **O1 blocks no defect fix.** The spike §7 splits this by axis and the split is
  correct: a ratio needs a denominator, an 89%-of-wall single-threaded host phase
  does not.
* **[#655](https://github.com/mudler/vllm.cpp/issues/655) blocks every ratio and
  is not this row's to close.** Recorded, not taken.

## Work breakdown

Each stage lands on its own branch, is reachable from a production entry point at
its own merge commit, is revertible without touching the next, and **owes an open
issue before it starts**. Every issue this campaign stages is named below exactly
once.

### The order, argued

**The instrument is not first because instruments are virtuous. It is first
because the ranking has three independent defects and all three are measurement
defects.** (a) The evidence is unretrievable — the sampler CSVs, rung 2's
`run.log` and `render8-console.log` exist only on a host that stopped answering
(#1040), and the spike marks even its *passing* numbers with that reason. (b) The
subject is unnamed — #1087 measures a 1731 s phase and says in its own text *"Do
not guess it from the duration"*, and #1024's GPU-zero window is now known not to
be the denoise (§*Our baseline* finding 1) and known not to be the decode. (c)
The ranking is **stale**, not merely unverified: #1009 and #1208 both landed
after it and neither was re-measured (finding 4). A campaign staged on that
ranking would be spending months against a profile that no longer exists.

**After W1, the order is not by magnitude. It is by what each stage does to the
campaign's own measurement loop.** W2 and W3 are small, cheap and revertible, and
they remove the two things that make every later measurement expensive or
dangerous: the memory headroom that decides whether a render completes on a
119 GiB box at all, and the 7.5 minutes of staging paid at the front of every
render *and every gate run*. Buying those first makes W4 and W5 cheaper to
iterate, and W5 is the stage that will need the most iterations. That is the
whole argument, and it is a scheduling argument, stated as one.

**W5 is last of the implementation stages despite being ranked first on
magnitude, and that is a deliberate disagreement with the spike's table.** Lever
1 is ranked first on FLOPs — 7.25 TFLOP of dense 3x3x3 convolution — and that
arithmetic is sound. But the evidence #1024 produced cannot support it: the
sampled GPU-zero window had written no frame, so it was not in the decode. The
spike itself reaches the same conclusion from the other side, recommending *"6,
2, 3, then 1"* and then *"6, 3, then 1"*. W5 stays in the campaign, ranked on its
arithmetic, and starts after the stage that tells us what the render is actually
doing. **If W1's phase table puts the decode back on top, W5 moves up and this
section is amended** — that is what W1 is for.

### The stages

| Stage | Issues | Lands | Size |
|---|---|---|---|
| **W0** the instrument | [#1010](https://github.com/mudler/vllm.cpp/issues/1010), [#1040](https://github.com/mudler/vllm.cpp/issues/1040) (artifact contract half) | phase-boundary timestamps and per-phase peak host/device bytes, emitted from the render path on the shipped default; the evidence-path contract in §*Risks/decisions* H2 | small |
| **W1** re-derive the ranking | [#1040](https://github.com/mudler/vllm.cpp/issues/1040) (closes), [#1024](https://github.com/mudler/vllm.cpp/issues/1024) (re-measured, with the control it owes) | no code. A phase table at two geometries on an idle leased box, artifacts in the repo, `## Now` amended with the re-derived order | small, one lease |
| **W2** the load path | [#1016](https://github.com/mudler/vllm.cpp/issues/1016), [#1015](https://github.com/mudler/vllm.cpp/issues/1015), [#1210](https://github.com/mudler/vllm.cpp/issues/1210) | three independent commits on one branch, each with its own red-first RSS or fuse-count case | small each |
| **W3** the staging path | [#1021](https://github.com/mudler/vllm.cpp/issues/1021) | batched/overlapped device staging behind a byte-compare | medium |
| **W4** the conditioning stage | [#1269](https://github.com/mudler/vllm.cpp/issues/1269) | a device weight arm for the Gemma-4 tower and the caption projection, and the queue threading that reaches it | medium-large |
| **W5** the video VAE decode | [#1007](https://github.com/mudler/vllm.cpp/issues/1007), [#1011](https://github.com/mudler/vllm.cpp/issues/1011) as a rider | a `vt::` conv3d op with CUDA and CPU arms, a device `Ltx2VaeWeights`, and the memory format the arm consumes | large |
| **W6** the residency question | [#1014](https://github.com/mudler/vllm.cpp/issues/1014) | attribution or a documented non-reproduction | medium, conditional |
| **W7** the graph decision | — ([#1164](https://github.com/mudler/vllm.cpp/issues/1164) is decided, not implemented, here) | a measurement and a verdict | small |
| **O1** the oracle record | [#1012](https://github.com/mudler/vllm.cpp/issues/1012) | `diffusers` recorded as covering LTX-2.5, with a gateability measurement | small-medium |
| — | [#1009](https://github.com/mudler/vllm.cpp/issues/1009) | **CLOSED.** Landed as [#1041](https://github.com/mudler/vllm.cpp/issues/1041); `ltx2_video_vae.cpp:170`, `:218`, `:276`. Listed so the count closes, not staged | — |
| — | [#1202](https://github.com/mudler/vllm.cpp/issues/1202) | **DEFERRED behind W1**, argued below | — |

### Two levers this campaign does not carry as stages, and why

**[#1011](https://github.com/mudler/vllm.cpp/issues/1011) does not get a stage. It
becomes a rider on W5.** The spike §4 already re-ranked it from *"the 60 GiB"* to
byte traffic and memory format. Once W5 puts the decode on the device, upstream's
memory-efficient **host** decode is an optimization of a path production no longer
takes, and its one genuinely portable part — the memory format — is the thing that
decides which kernel family W5's device arm can reach. Carrying it as a separate
stage would schedule a second independent rewrite of `CausalConv3d` against a host
arm we are trying to stop using. **If W5 is refuted** — no device arm is
achievable, for a reason W5 names — #1011 comes back as its own stage against the
host fallback, and this paragraph is the record of that condition.

**[#1202](https://github.com/mudler/vllm.cpp/issues/1202) is deferred behind W1,
not dropped.** `Ltx2FuseLoraIntoTensor` is a scalar single-threaded loop at
~0.53 GFLOP/s and it should route through the `vt::` GEMM seam like everything
else. But the spike's own `## Owed` measures it at **2.3% of one pass in
10.4 min** on the full model, and it is paid only on a LoRA arm. Against W4's
39-100% bound that is small enough that ranking it before W1's table exists would
be guessing. It is deferred with a number attached rather than carried as debt
with no number, and W1's table decides it.

## Risks/decisions

### The hardware hazards, as gates

Each is a condition a stage's run script asserts, not a paragraph an operator is
expected to remember. A stage that cannot satisfy its applicable gate **does not
run**.

**H1 — a GB10 unified-memory OOM reboots the whole box, and the spike records
that it needs a physical power cycle.** Every stage that starts a render arms a
memory watchdog with a **named floor** before the engine starts, in the shape
`LTX25-RESOLUTION-ENVELOPE` used (a 2 s sampler with a stated `MemAvailable`
floor, which ended a run at 13.77 GiB against an 18 GiB floor rather than letting
the box fall over). **The watchdog's own arming is positive-controlled** before
the real run, because an absent hook looks exactly like an armed one and a broken
instrument fails toward a verdict about the code. Gate: no render starts without
a floor in its command line and a control that proves the watchdog fires.

**H2 — `/tmp` does not survive that reboot, and evidence has already been lost to
this.** `dgx.casa` runs an immutable Kairos OS: `/mnt` and the root overlay are
ephemeral and do not survive a reboot, while `/usr/local` is `COS_PERSISTENT` and
does ([`.agents/environment.md:390-398`](../environment.md),
[`nas-mount-path.md:29-40`](nas-mount-path.md)). Two rows already lost an `ncu`
comparison to a shared-box OOM-reboot
([`mxfp4-flash-audit-2026-08-06.md:33`](mxfp4-flash-audit-2026-08-06.md),
[`qwen3-mxfp4-flash-occupancy-2026-08-06.md:16`](qwen3-mxfp4-flash-occupancy-2026-08-06.md)).
Gate, three parts: (a) the run script **asserts its artifact path is not under
`/tmp`, `/var/tmp` or `/mnt`** and fails loudly if it is; (b) artifacts are copied
off the box **before the lease ends**, not after the run; (c) the copy is verified
by reading the file back, because `REMOTE_UNVERIFIED` is what
[#1040](https://github.com/mudler/vllm.cpp/issues/1040) is made of and unknown is
not success.

**H3 — the fleet is leased through `rc`, never through `ssh`.** `rc run` or
`rc hold` claims a device; raw `ssh` makes the fleet report the box free while
someone is on it, which is the contention the lease exists to remove
([`.agents/environment.md:10-31`](../environment.md)). Gate: every measurement
stage records its lease id, and a stage whose evidence carries no lease id is not
accepted. Run `rc devices` before calling any box unhealthy — `dgx.casa`
being unreachable to one session is not the same fact as the device being down.

**H4 — a lease carries bytes, not executables, and the submitting client's death
kills the job.** `/tmp`, `/var/tmp` and `$HOME` (`/home/rc`) inside a lease are
writable and take a real exec bit, so a runtime staged on `/workspace` runs after
a copy ([`.agents/environment.md:148-183`](../environment.md)); but `rc run`
submissions die with the detaching client, and this harness's own 10-minute
command cap will kill a queued job. Gate: a render stage stages its script where
the worker can read it and submits with a `--max-runtime` that covers the render,
and **never** holds a foreground client for the duration of a 2.5-hour run.

**H5 — a long A/B must not run a binary from a worktree.** Merging the branch
reaps the worktree mid-run and the measurement dies with it. Gate: every W1, W3,
W4 and W5 wall-clock run executes a binary copied outside any worktree, and
records the sha256 of what it ran.

### Decisions taken here

**Row state is `READY`, not `SPIKE`.** A `SPIKE` produces a finding and ships no
code, which is precisely what
[`ltx25-decode-speed.md`](ltx25-decode-speed.md) already is and precisely what
[#1264](https://github.com/mudler/vllm.cpp/issues/1264) says is *not* missing. A
second spike over the same ground would be the duplication that issue's own
*"this issue does NOT duplicate it"* section forbids. This row has the nine
sections, an ordered staged plan, an issue per stage, and implementation
authority — which is the definition of `READY`. It is not `ACTIVE`, because no
stage has started and no gate has run; claiming `ACTIVE` on a spec-only commit
would be a lifecycle state with nothing behind it.

**No matrix row, and no claim file.** The four sibling `LTX25-*` rows
(`LTX25-DECODE-SPEED`, `LTX25-DECODE-THREADS`, `LTX25-TEXT-LINEAR-SEAM`,
`LTX25-RESOLUTION-ENVELOPE`) appear in **no** matrix, and the reason is
structural rather than an oversight: `scripts/check-agent-record.py` builds its
row-ID pattern from `MODEL`, `QUANT`, `KERNEL`, `BACKEND`, `ENG`, `KV`, `PAR`,
`SAMPLE`, `TOOLS`, `SPEC`, `SERVE`, `LORA`, `ATTN` and `LOAD` (`:327-338`,
`:599-603`), so an `LTX25-` id in a matrix is a row **no checker reads**, sitting
in a shared file that every future edit must then write. AGENTS.md names that
shape a lock. The record surfaces are therefore this spec, the append-only issue
index, and the one matrix cell this campaign genuinely invalidates —
`ENG-CUDAGRAPH-DIFFUSION` in [`engine-matrix.md`](../engine-matrix.md), whose
blocker list names #1024, #1007, #1087 and #1010 and now names the row that owns
their unblock order. No claim file is written, because the row is unclaimed by
design: each stage goes to its own fresh implementer, and a claim naming this
session would be stale at merge.

**No `docs/STATUS.md` or `docs/BENCHMARKS.md` edit.** This change ships a spec
and issue rows. No capability changes lifecycle state — LTX-2.5's line at
`docs/STATUS.md:146` describes L1-L9c as landed and this row lands nothing — and
no measurement is taken, accepted, or made pending. The spike §8 argued the same
for the same reason and the projection table in AGENTS.md agrees: editing
`.agents/` on its own owes neither. **W1 owes a `docs/BENCHMARKS.md` edit** when
its phase table lands, and W5 owes one when its wall is accepted.

### Risks to the plan itself

* **W1 needs a lease on a box that has been unreachable.** If `dgx.casa` does not
  answer, W1 reports `BLOCKED` with `REMOTE_UNVERIFIED` and the campaign runs W2
  **and W3** and then stops — the two load-path defects and the staging path are
  provable on any box with the checkpoint, because their gates are RSS and
  byte-compares, not wall clock. W4 and W5 do **not** start on an unreproduced
  ranking. That is the stop condition, and it is deliberate: proceeding would
  repeat the exact failure #1040 records.
* **W5 may be large enough to need its own campaign.** If its port map comes back
  with more than one new `vt::` op family, the stage splits and this section is
  amended rather than the stage being pushed through as one review.
* **The tolerance for W4 and W5 is not yet argued.** Both compare against a host
  arm rather than an upstream oracle, and a tolerance chosen after seeing the
  numbers is not a gate. Each stage argues its tolerance **at the site, before
  the run**, in the shape [`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) §3
  used.
* **This campaign's own numbers can go the way the spike's did.** H2 is the
  answer, and it is written as an assertion in a run script rather than as an
  instruction to a person, because the instruction form has already failed once.

## Owed

| Issue | Stage | State |
|---|---|---|
| [#1264](https://github.com/mudler/vllm.cpp/issues/1264) | this row: the staged campaign spec | closed by this row landing |
| [#1440](https://github.com/mudler/vllm.cpp/issues/1440) | W0 (gate) | **closed by the third review's repair.** Four of the containment case's five assertions were ratios against the sub-scope anchor, so an anchor that moved with its leaf satisfied all of them; the repair adds a record COUNT taken from the render's own counters and a rule that nothing but an anchor may be emitted `nested`. See `### What a THIRD fresh review found` |
| ISSUE-GH-1010 | W0 | **closed, for a run that FINISHES.** The render writes a phase table on the shipped default and the ABI names it through `vllm_video_last_phase_log` (v23). Read the row below it before quoting that as "the render is instrumented" |
| [#1413](https://github.com/mudler/vllm.cpp/issues/1413) | W0-live | **closed by `## W0-live` below.** W0's table is written by the success path only, and nothing at all is emitted while a render runs, so an aborted render and a working one both report nothing and the ~162 s DiT forward of [#1375](https://github.com/mudler/vllm.cpp/issues/1375) has no in-process counter |
| **a table on a run that does NOT finish** | [#1413](https://github.com/mudler/vllm.cpp/issues/1413), stage W0-live | **the LIVE half is closed by this change; a PARTIAL TABLE on abort is still owed and has no issue.** `WritePhaseLog` has exactly two call sites (`ltx2_video.cpp:2259` audio-only, `:4677` video) and both sit immediately before a successful `return`, three lines after `im.trace.completed = true`. Nineteen `VT_CHECK` sites throw out of `Ltx2VideoEngine::Generate` above them and that body contains no `try` and no `catch`; `vllm_video_generate`'s own two catches set an error and return, and `engine->last_phase_log` is assigned on the success path only. **So a render that is killed, aborted by a lease governor, refused by a guard, or still running leaves no `phase-log.json` at all** — not a truncated one, not an empty one, nothing. The mutation is the demonstration: deleting the video call site removes the file entirely and the W0 gate goes red on `REQUIRE(probe.good())`. That matters here more than anywhere, because the runs this campaign has are the ones that died: [#1375](https://github.com/mudler/vllm.cpp/issues/1375) is `child exit=-15` at 0 frames, and [`ltx25-decode-speed.md`](ltx25-decode-speed.md)'s two rungs are `EXIT=137` and `EXIT=1` at 0 frames. A reader who takes "the render writes a phase table" at face value will expect a 2.5 h render that is killed at 2.4 h to leave a table naming where it was. It leaves none. #1413 CLOSED the live half, in `## W0-live` below — a line per phase boundary and per DiT forward, so a killed run is legible from its stderr; a signal handler that flushed a PARTIAL table on abort is a separate change with its own re-entrancy argument and is not owed by either |
| the phase table's DEVICE column | W1, and it needs [#1126](https://github.com/mudler/vllm.cpp/issues/1126) first | **owed, and a LEASE WILL NOT CLOSE IT.** The column is defined as the driver's live in-use bytes, read per phase through `vt::Backend::DeviceMemoryInfo`. It reports the `-1` no-probe sentinel in W0's artifact, and there are TWO reasons stacked, only one of which is a scheduling problem. **(1)** The render W0 could take was the CPU arm, where the sentinel is correct — `dgx:gpu0` was busy with two queued jobs and `orin:gpu0`, the one free device, holds no LTX-2.5 checkpoints. **(2)** The one that matters: **`CudaBackend` does not override `DeviceMemoryInfo` at all**, which is [#1126](https://github.com/mudler/vllm.cpp/issues/1126), and `grep -rn 'DeviceMemoryInfo' src/vt include/vt` returns exactly the base declaration at `include/vt/backend.h:94` and one override, `src/vt/rocm/rocm_backend.hip:373`. So a CUDA render on `dgx:gpu0` would print `-1` in every row of this column too, and it would print it for a reason no lease can fix. **What the column cannot report today, stated as three things:** how many device bytes the DiT staging leaves resident; whether the denoise grows device residency across steps; and whether the ~59 GiB #1014 asks about is device-class at all. **Why W0 did not just wire it:** `include/vllm/platforms/interface.h:68-72` records that CUDA's absence from that seam is load-bearing — `Gemma4MoE`'s device-expert LRU is the seam's only consumer and is DEAD on CUDA. Narrowed, because the broad form of this sentence is wrong: the bf16 arm `EnsureGemma4Fp8ExpertOnDevice` (`gemma4_moe.cpp:548`) is dead for a SECOND, independent reason — it refuses at `:571` on `vt::HasMatmulBTAlphaBeta`, implemented only by ROCm ([#1205](https://github.com/mudler/vllm.cpp/issues/1205)) — so the override alone would not wake it. The arm the override WOULD wake is the FP8-native one, `EnsureGemma4Fp8NativeOnDevice` (`:611-628`), which reaches `MakeRoom` behind no device gate but the probe, with a 2048 MiB fill-only budget by default (`:416-432`). One named arm, not every CUDA model — and still a behaviour change with its own measurement. That is #1126's change to make, not an instrument's. **What W1 does instead, until #1126 lands:** sample `nvidia-smi --query-compute-apps=used_memory` per phase beside the table — the fallback this spec's W6 row already names, and the one instrument GB10 answers, since `--query-gpu=memory.used` returns `[N/A]` there. On GB10 the peak HOST column is not a poor substitute either: the pool is unified, so host resident bytes and device bytes are the same 119 GiB arena, and that column does report |
| ISSUE-GH-1040 | W0 (contract) + W1 (closes) | contract half **met** — the table is a file beside the frames rather than a console line, and it is retrievable from this repo at `benchmarks/demo/`. The closing half is W1's |
| ISSUE-GH-1024 | W1 | owed; its `utilization.gpu` positive control is still unrecorded in this tree |
| ISSUE-GH-1016 | W2a | owed |
| ISSUE-GH-1015 | W2b | owed; host arm only, see `## Our baseline` finding 3 |
| [#1210](https://github.com/mudler/vllm.cpp/issues/1210) | W2c | owed |
| ISSUE-GH-1021 | W3 | owed |
| [#1269](https://github.com/mudler/vllm.cpp/issues/1269) | W4 | owed; filed by this row |
| ISSUE-GH-1007 | W5 | **closed by `## W5` below, for the CONVOLUTION.** `vt::OpId::kConv3d` exists with a CPU and a CUDA arm, and the decode dispatches every `nn.Conv3d` call through it on the queue the engine resolved. Read the two rows under it before quoting that as "the decode runs on the device" |
| [#1451](https://github.com/mudler/vllm.cpp/issues/1451) | W5 | **owed, filed by W5.** The arm is CONVOLUTION-ONLY. Every buffer between the convolutions — the norms, the SiLU, the AdaLN, the noise injection, `DepthToSpaceUpsample`, `AttnBlock3d`, `Linear3d`, `unpatchify` — is still a host loop, and there is no device `Ltx2VaeWeights`, so a non-CPU queue pays an upload and a download per convolution and re-uploads each weight. Upstream never moves the tensor back (`single_gpu_model_builder.py:267-288` at `:273`, `blocks.py:1139`; vLLM-Omni `interface.py:92` "VAE(s) (always on GPU)"), so this is a divergence in memory behaviour and not only a cost. **No number is attached** — W5 took no lease |
| [#1452](https://github.com/mudler/vllm.cpp/issues/1452) | W5 | **PARTLY CLOSED on 2026-08-25: the arm now RUNS, and the bit-identity is MEASURED rather than argued. Two of the four owed items remain.** The four were a compile, a `memcmp` CPU-vs-CUDA arm, an end-to-end pixel comparison, and the f16/bf16 storage the arm refuses by name. **(1) The compile: MET, and stated more narrowly than the first draft of this row stated it.** That draft said "first compile ever", and that is FALSE: `cuda-fat-build` (`.github/workflows/ci.yml`) builds this translation unit on every pull request in an `nvidia/cuda` container, and the workflow records a **measured 123.0-minute finish** against its 180-minute timeout, so the job completes and its verdict does land. #1452's own text is the careful version and this row flattened it. What IS new: a compile for **`sm_121a`** with the emitted object inspected — nvcc **13.0.88**, `-DVLLM_CPP_CUDA_ARCHITECTURES=121a`, Release, in an `rc` lease on `dgx:gpu0` (GB10), **zero errors and zero warnings**, and `cuobjdump --list-elf` reporting one ELF, `cuda_conv3d.cu.1.sm_121a.cubin`, so the object carries real `sm_121a` code rather than PTX alone. `src/vt/cuda/cuda_ltx2_vae.cu` compiled clean in the same tree and run; it was PR #1909's unmerged head at the time and landed on `main` as `fc2c5be23` while this branch was in review, so that verdict now applies to `main` directly. **(2) The `memcmp` arm: MET, and this is the genuinely new half — the first EXECUTION of the kernel on any device.** `tests/vt/test_ops_conv3d.cpp` gains three cases in the shape `test_ops_conv1d_general.cpp` ships for `kConv1d`. On the device: **7 cases / 2071 assertions green**. **That the CUDA provider really ran is held by the dispatch, not by the log line.** The `vt::Conv3d` announcement fires BEFORE `GetOp`, so it proves only that the queue was non-CPU; what proves the provider is that `GetOp(kConv3d, kCUDA)` THROWS when no CUDA arm is registered, and that the portable CPU tier cannot install onto `kCUDA` because `CudaBackend` leaves `DeviceMemoryIsHostAddressable()` false (pinned by the `static_assert` at `src/vt/cuda/cuda_backend.cu:362-368`). A CPU-vs-CPU comparison is therefore unreachable here rather than merely unlikely. **The gate was mutation-proved on the same hardware in the same run**: rewriting the kernel's `tap = __fadd_rn(tap, __fmul_rn(x, w))` as plain `tap += x * w`, exactly the FMA contraction the file's header says it refuses, turned it **RED at 2 of 7 cases and 15 of 2071 assertions**, and a byte-for-byte restore returned it to 7/7 and 2071/2071. **The first draft of this row said only 4 of the 15 shapes moved, and that was read off a `tail -40`-TRUNCATED log rather than counted.** `RequireBitIdentical` emits two assertions per call and a moved shape fails exactly one, so 15 failed assertions is **14 of the 15 shapes plus the cancellation case** — the shape table is a far better instrument for FMA contraction than that draft credited, and the correction runs the opposite way to the story it was told in. The cancellation case remains the one that would catch a pure REASSOCIATION, which this mutation is not; its own teeth check is verified separately, and removing the shared weight row drives it red. The data those shapes run on is `Spread()`, ±64 across ~6 decades — NOT `Rng::Uniform`'s [-1, 1], which only the four pre-existing CPU cases use and which an FMA mutation of the CUDA kernel cannot reach. **(3) The end-to-end pixel comparison: STILL OWED**, and now with a cost. The staged lease kit at `/mnt/nas_share/rc/ltx25-fullmodel/` carries 4 of the 6 checkpoints its own `render.sh` references; the spatial upsampler and the distilled LoRA are on the NAS but outside the `rc/` subfolder a leased worker mounts, so they need staging first. The comparison also needs BOTH arms of a 22 B DiT render, and the host arm is the expensive one. No wall-clock was taken for either and none is claimed. **(4) The narrow dtypes: STILL OWED, but PINNED.** `conv3d: the CUDA arm refuses the narrow dtypes BY NAME` asserts the refusal on the device, so the gap cannot change silently while it waits for a kernel. **Two riders handled in the same change:** `src/vt/ops.cpp`'s first-non-CPU-dispatch notice claimed "This arm has never been run on real hardware", which this run made false — NARROWED rather than deleted, since for every other accelerator type the sentence is still true. And the three new cases `[SKIP]` on a CPU-only box, where `ctest --output-on-failure` discards their stdout and doctest scores them `passed`; every CI lane here is CPU-only, so they gate nothing off a leased device. That is a standing property of this suite and of `test_ops_conv1d_general.cpp`, recorded here rather than overstated in a comment |
| [#1939](https://github.com/mudler/vllm.cpp/issues/1939) | W5 | **owed, filed in flow by the fresh review of PR #1938 and NOT fixed there.** The three CUDA cases #1452 adds, and the `kConv1d` ones `test_ops_conv1d_general.cpp` has carried since `vt-conv1d-f32-accumulator`, all `[SKIP]` without a CUDA backend — and under `ctest --output-on-failure` doctest scores a skipped case as PASSED and never prints the line, because an early `return` is indistinguishable from a case with no failing assertion. No CI lane here owns a GPU and nothing under `scripts/` or `.github/` greps for `[SKIP]`, so on every automated run those cases report a pass having exercised nothing, and the #1452 measurement lives only as a hand-taken lease result. Two riders in the same issue: the cancellation case's teeth check is count-based and does not bite when the MAGNITUDE is lost (`kBig = 1.0` keeps `differing` at 2266 while the maximum absolute difference falls from 9275.17 to 9.8e-4), and `RunCaseF32` leaks on a throw exactly as the `Stage()` helper it was copied from does |
| [#1471](https://github.com/mudler/vllm.cpp/issues/1471) | W5 | **owed, filed in flow by W5's fresh review (F7) and NOT fixed here.** `vt::Conv2d` (`src/vt/ops.cpp:2749-2750`) and `vt::DepthwiseConv1d` (`:2883`) carry the identical truncating output-extent expression that F7 fixed in `vt::Conv3d`, so both disagree with the torch shape contract they advertise when the span is negative and stride > 1. Fixing them means two more red-first cases and a fresh review over ops this row does not own, and no caller is yet known to reach either path — that audit is part of the issue. `Conv1dOutLength` and `ConvTranspose1dOutLength` already guard it correctly, so the right shape is in the same file |
| ISSUE-GH-1011 | W5 rider | owed. W5 did not take it: `[C, T, H, W]` is what both arms consume, and the memory format only decides which kernel family a RESIDENT arm can reach, which is [#1451](https://github.com/mudler/vllm.cpp/issues/1451) |
| [#1014](https://github.com/mudler/vllm.cpp/issues/1014) | W6 | owed, conditional |
| [#1012](https://github.com/mudler/vllm.cpp/issues/1012) | O1 | owed |
| ISSUE-GH-1202 | deferred behind W1 | owed, with a number: 2.3% of one pass |
| [#1439](https://github.com/mudler/vllm.cpp/issues/1439) | W0 (its own gate) | **REPAIRED but NOT CLOSED, by the `#1439 repaired` section below: the prologue is now named `load.open`, so the time is measured rather than tolerated, and the assertion is byte-for-byte unchanged. It stays OPEN because the repair was measured on a box at loadavg 24-31, where the render's wall is 7-10 s and BOTH arms sit near 99.96% — the flattering regime this issue itself identified. What is owed is one quiet-box run at fixture scale, in the 0.22-0.58 s regime where the floor actually bites. What follows is the record of why it was red.** **It was RED on `main` rather than on any branch.** `CHECK(leaves >= 0.95 * wall)` in `ltx2 video: a render through the ABI emits a phase table that SUMS to wall` is a RATIO, and the un-named residue is 4.80% to 6.32% of `wall` across twelve runs on one x86 box - so the 95% floor sits INSIDE the measurement's own range at the 64x64 / 9-frame fixture scale, and the case decides by coin flip, mostly red. With this lane's four files reverted so the binary is main at `89261c955`, six runs read 94.32%, 95.20%, 93.74%, 94.20%, 94.69%, 94.19%; the W0-live merge reads 93.82%, 93.68%, 94.34%, 94.62%, 94.39%, and 94.12% with `VLLM_RENDER_PROGRESS=0`, which exonerates the live emitter. Box contention is NOT the cause and main's one green disproves it: that run had `wall=0.579684s`, more than double the others, because the box was loaded - a SLOWER render passes. The tolerance was argued for the 21.004 B render, where the residue is a far smaller fraction of the wall. Naming the un-named time, or bounding `unaccounted_seconds` beside the ratio so the assertion says the same thing at both scales, is gate semantics and owes its own row |
| [#1727](https://github.com/mudler/vllm.cpp/issues/1727) | (1c), sixth shape | **owed, filed by the fresh review of PR #1712, and a CONSTRAINT ON A RE-PROPOSAL rather than a live defect: the sampler it is about is NOT in this tree.** Two of the boundary sampler's three terms are unfalsifiable by the gate -- deleting the flushed write leaves the estimator inside its own scatter, green 3 of 3 -- and the estimator is unmeasured in two regimes that decide it: the sanitizer lanes, and the FULL 102-case binary rather than a `-tc=` filter, which is where #1576's 171x swing was measured |
| [#1728](https://github.com/mudler/vllm.cpp/issues/1728) | (1c), sixth shape | **owed, filed by the fresh review of PR #1712 as a HYPOTHESIS with no measurement behind it, and about a sampler that is NOT in this tree.** On a ROCm runner the 1 kHz boundary sampler would drive `hipMemGetInfo` through `SampleLocked`'s device probe, under the process-wide phase mutex, for the whole case. `CudaBackend` does not override `DeviceMemoryInfo` (#1126) so the CPU and CUDA lanes are unaffected; `src/vt/rocm/rocm_backend.hip:373` does. No GPU lease was taken |
| [#1164](https://github.com/mudler/vllm.cpp/issues/1164) | W7 decision point | not owed here — owned by `ENG-CUDAGRAPH-DIFFUSION`; this row owns its unblock order |

**ISSUE-GH-1009 is not owed. It
landed**, as [#1041](https://github.com/mudler/vllm.cpp/issues/1041), and listing
a closed issue as owed by a row that did not close it misattributes the work and
leaves a fixed defect reading as open debt — the same argument the spike makes
for #1008 and #1208.

Also owed, and not attached to a stage:

* **[#655](https://github.com/mudler/vllm.cpp/issues/655)**, the `ltx_core`
  oracle registration. It blocks every ratio and it is not this row's to close.
* **The GB10-to-x86 per-core ratio** that bounds #1269 to 39-100% rather than a
  number. Owed to [`ltx25-text-linear-seam.md`](ltx25-text-linear-seam.md)
  `## Owed`, and W1's lease could close it in the same run.
* **The third single-core stretch** of the #1208 trace (2589 s+, RSS flat at
  31 GiB), unattributed. W1's phase table should name it; if it does not, that is
  a W0 gap and W0 iterates.
* **A VAE-SIDE sub-scope for `decode.video`.** The anchor W0 lands sits in the
  driver: `decode.video.chunk` runs from the leaf's own open to the moment
  `Ltx2VideoDecodeStreaming` hands a chunk back. That END is a production event
  and it is what catches M10, but the WORK it wraps is the whole call rather than
  the tile accumulation inside it. The honest sub-scope is around
  `AccumulateTemporalGroup`
  (`src/vllm/model_executor/models/ltx2_video_vae_tiled.cpp`), which is outside
  the authority W0 was dispatched with. Owed to W5, whose lever this phase is,
  and it is a refinement of a gated phase rather than an ungated one. **W5 did
  NOT take it, and says why in §W5.1:** it is an instrument refinement rather
  than a device arm, and
  [#1439](https://github.com/mudler/vllm.cpp/issues/1439) has the phase table's
  own sum gate RED on `main` at fixture scale, so adding leaves under it while
  that ratio assertion is coin-flipping would make two changes
  indistinguishable. It stays owed.
* **The video ENCODER's queue.** `CausalConv3d` is shared by the decoder and the
  encoder, so W5 routed all nine call sites and the encoder's convolutions
  dispatch through `vt::Conv3d` too — but only ever on the CPU queue, because
  `Ltx2ConvVideoEncode` was not given a queue parameter. That is scope, not an
  oversight: the encoder is not on the render path this campaign measures. It has
  no issue, and it is one parameter wide.

## Outcome — W0, the instrument

Landed on `row/LTX25-RESIDENCY-W0`, issue
[#1010](https://github.com/mudler/vllm.cpp/issues/1010), base `5f68e60df`.

### The gate, and the number

**PASS.** One completed render through the `vllm.h` video ABI emits a phase
table whose named leaves sum to **99.94% of wall** — 4.459892 s of leaves
against 4.462572 s of wall, residue **2.68 ms** over 33 entries and 118 samples.
The tolerance was fixed at **>= 95%** in the test's own comment and in the
red-first commit message *before* the instrumented run, and the sum is checked
in the same case that checks the named boundaries, because one leaf called
`render` would sum to wall exactly and measure nothing.

The same case has produced **98.33% at a 0.158 s wall, 98.74% at 0.147 s, 99.88%
at 1.676 s, 99.94% at 4.463 s, 99.96% at 6.138 s and 99.84% at 12.030 s** on six
runs of the identical binary at the identical geometry. The RATIO is what the
gate reads and it never came near the 95% floor; the WALL moved by a factor of 82
across those runs because the box was building other sessions' trees — and the
0.147 s and the 4.463 s are ONE MINUTE apart, which is the sharpest form this
finding has taken. That is why no wall figure in this section is a benchmark and
why W1 is written as a lease on an idle box.

The evidence file is
[`benchmarks/demo/ltx25_phase_log_fixture_cpu.json`](../../benchmarks/demo/ltx25_phase_log_fixture_cpu.json),
sha256 `59a860163e6ab2569c789209be709858915ba58393e8eb6900559f192ea7b950`.
**Re-taken twice.** First for the first review's F4, which found the original
artifact carrying no provenance at all in a directory whose every sibling carries
`_source`, `hardware`, `workload` and `footnotes`; then for the second review's
finding above, because an artifact taken before the `denoise.step` and
`decode.video.chunk` anchors existed cannot show the containment the gate now
requires. Every phase record in it is the render's
own output; the four keys above them — `_source`, `_caveat`, `_headline`,
`_footnotes` — are written by hand at commit time, because a render cannot know
its host, its checkpoint or what else the box was doing, and those are exactly
the facts whose absence made #1040 and #1087 unreadable. The library now writes
the part it CAN know into every phase log it produces: `notice`, `sum_rule` and
`sampler_enabled`. Provenance:

| | |
|---|---|
| producer | `examples/ltx2_gen` (`build/examples/ltx2-gen`), through `vllm_video_engine_load` + `vllm_video_generate` |
| tree | binary built from `3dc2ae98b` on `row/LTX25-RESIDENCY-W0` |
| build | `cmake -DCMAKE_BUILD_TYPE=Release -G Ninja`, gcc 13.3.0 |
| host | `mudler-ubuntu-box`, Linux 6.8.0-136-generic x86_64, **20 cores and CONTENDED** — 1-minute load average between **31 and 55** across the render and the one a minute before it, other sessions building throughout |
| checkpoint | the reduced-dimension fixture `tests/vllm/multimodal/ltx2_video_fixture.h` writes, in the shipped file format |
| geometry | `--frames 9 --width 64 --height 64 --seed 7 --max-phase 0 --device cpu` |
| completion | 9 frames written, **9 distinct per-frame md5s**, plus a 48 kHz WAV. The exit code is not the completion gate here ([#1149](https://github.com/mudler/vllm.cpp/issues/1149)); the distinct md5s are |
| device column | **every row reads `-1`.** See `## Owed` — this is the CPU arm, and `CudaBackend` would read `-1` as well |

**The wall figures are NOT a benchmark and nothing may quote them as one.** The
host was contended, the checkpoint is a reduced fixture, and the same case has
measured 0.15 s, 0.16 s, 1.68 s, 4.46 s, 6.14 s and 12.03 s of wall on six runs
of the identical binary. What the artifact supports is the SHAPE of the table and the fact that
its parts add up. `docs/BENCHMARKS.md` is therefore untouched, which is what
this spec's `### Decisions taken here` already said W0 owes: W1 owes the
benchmark edit, on a leased idle box, at two geometries.

### What the table says about this render, which is not what anybody expected

Two leaves are the whole render on this fixture, and **which of them is larger
changed between two runs of the identical binary**. On the artifact this row
first shipped, `denoise` was 8.127 s (67.7% of the named leaves) against
`decode.audio` 3.062 s (25.5%). On an earlier run of the same case at the same
geometry the order was reversed: `decode.audio` 0.0889 s against `denoise`
0.0539 s of a 0.158 s wall. The first re-take recorded a third split again —
`denoise` 48.3% against `decode.audio` 42.6% — and **the artifact this row now
ships records a fourth**: `denoise` 75.79% against `decode.audio` 18.07%. The
gap between the two has been 2.7x, 0.6x, 1.1x and 4.2x on one binary at one
geometry.

The re-take also names something the first artifact could not, because F1 split
it: **`decode.audio.vocoder` alone is 0.8006 s, 17.95% of the named leaves, and
99.998% of `decode.audio`.** Almost the whole of `decode.audio` is the vocoder,
and until this row that phase had no name of its own anywhere in the tree. The
same is now true one level up: `denoise.step` covers **99.993%** of `denoise`
over 8 denoiser evaluations, and `decode.video.chunk` **99.982%** of
`decode.video`.

That reversal is worth more than either number. The host was building other
sessions' trees throughout, and the two phases do not have the same threading —
so a contended box does not merely add noise to a phase table, it can change the
RANKING the table reports. This campaign exists because a stale ranking was
taken as a profile, and W1's own gate is an idle leased box for exactly this
reason. Nothing here says anything about the 21.00B checkpoint, whose DiT load
alone is minutes; what it says is that the table is now capable of showing an
ordering, and that reading one off a contended run is how the last ranking went
wrong. That ordering belongs to a two-block reduced DiT and says
nothing about the 21.00B checkpoint — it is recorded because it is the first
time any LTX-2.5 render in this tree has said where its time went at all, and
because it is exactly the kind of term the spike's lever table could not have
ranked.

### What was rejected, and why

**A shared phase table with the process as its scope was rejected**, after it
shipped and failed. `Begin` was first written idempotent, so a process that
loaded several engines accumulated one timeline and reported the time between
unrelated renders as this render's residue: the 90-case suite read 8.33 s of
residue against a 5.20 s budget on a render that had accounted for 99.96% of its
own wall. A load now discards the earlier timeline. The origin still sits at the
LOAD and not at the generation, because the spike measures ~7.5 minutes of DiT
staging paid at the front of every render.

**Chaining the phases into a partition was rejected.** A `Mark(name)` that
closed the open phase and opened the next would make the sum hold by
construction, and the missing time would be silently charged to whichever phase
happened to be open. The scopes therefore leave real gaps, and the residue is a
real quantity — which is what makes the gate refutable at all, and what mutation
M2 below demonstrates.

**Appending a field to `vllm_video_result` was rejected** in favour of the
`vllm_video_last_phase_log(engine)` query. Growing an OUTPUT struct is the one
append a caller cannot absorb by zero-initializing, because the library writes
the new field using its own `sizeof`.

**Instrumenting `vt`'s allocator for the device column was rejected** as a
different row's change: it is a hot path taken thousands of times per step. The
column reads `vt::Backend::DeviceMemoryInfo` instead, which is backend-neutral
and is the shape a device-byte question already has in this tree.

**Overriding `CudaBackend::DeviceMemoryInfo` so the column would answer was also
rejected, and this is the one rejection a reader should not skip.** It would
have made W0's own gate look complete on a CUDA box, and it is
[#1126](https://github.com/mudler/vllm.cpp/issues/1126)'s change:
`include/vllm/platforms/interface.h:68-72` records that the seam's only consumer,
`Gemma4MoE`'s device-expert LRU, is DEAD on CUDA. **This row first wrote that as
"wakes a landed residency policy on every CUDA model", which claims more than the
code supports and is corrected here.** The bf16 arm
`EnsureGemma4Fp8ExpertOnDevice` (`gemma4_moe.cpp:548`) is separately dead behind
`vt::HasMatmulBTAlphaBeta` at `:571` — ROCm holds the only implementation
([#1205](https://github.com/mudler/vllm.cpp/issues/1205)) — so the override
would not wake that one. It would wake `EnsureGemma4Fp8NativeOnDevice`
(`:611-628`), which reaches `MakeRoom` with the probe as its only device gate and
a 2048 MiB fill-only default budget (`:416-432`). One arm, named, and still a
behaviour change with its own measurement — which is the reason, and the reason
did not need the overstatement. An instrument that changes model behaviour to
make its own column non-empty is not an instrument. The column reports `-1` and
`## Owed` says why.

### Mutations, each with its diff, its compile status and its exit code

| # | Mutation | `git diff --stat` | compile | focused gate |
|---|---|---|---|---|
| M1 | delete the `WritePhaseLog` call in `Generate` — the **reachability** mutation | 1 file, +2/-2 | rc 0, 0 errors | **RED**, exit 1, 4 assertions, 1 failed |
| M2 | mark the `denoise` scope a span so its time is named by nobody | 1 file, +1/-1 | rc 0, 0 errors | **RED**, exit 1, 273 assertions, 1 failed, on `leaves >= 0.95 * wall` (0.078 s of 0.117 s) |
| M3 | make `HostResidentBytes` return `-1` | 1 file, +4/-1 | rc 0, 0 errors | **RED**, exit 1, 273 assertions, **21** failed, one per phase entry |

M1 is the mutation `.agents/reachability.md` asks for: the production call site
is deleted in a scratch copy and the focused gate goes red, so the gate measures
a capability rather than a class.

### What a fresh review found, and the mutations that closed it

A fresh reviewer returned `PASS_WITH_FINDINGS`. It reproduced M1, M2 and M3
exactly, confirmed the instrument is reachable on the shipped default, and found
no correctness defect. Its central finding is the one that mattered, because W0
gates a campaign: **this gate could not tell a correct table from a useless one,
and the reviewer built the useless one.**

**F2 — existence plus a sum is not attribution.** Mutation M4 leaves the
`decode.video` leaf open across the audio decode and gives `decode.audio` its
name with no work beneath it. All six required names are present, no leaf nests,
and the leaves account for 99.9% of wall — so the gate passed, over a table that
reported the video decode as 32% of the render (it is 2.4%) and the audio decode
as free (it is a quarter of it). A W1 reader ranking levers off that sends W5,
this campaign's largest stage, at the wrong phase and drops the audio decode
entirely. Seven of the 21 leaves also read 0.0000 s on the gate's own render, so
nothing about their placement was proven either.

**The suggested repair was a differential over two frame counts, and this box
refuted it.** Render at 9 frames and at 33, and require the phases whose work
scales with the clip to grow. It was written, built and run here. The 9-frame
render measured **8.03 s** of named leaves and the 33-frame render **3.80 s**,
minutes apart on the same binary — the 2.5x longer clip cost HALF the time — and
an earlier run of the same 9-frame render measured **4.80 s**. Wall noise here is
a factor of two in both directions against a 2.5x signal, so a
seconds-differential is a coin flip wearing a gate. The form is dropped and the
measurement is kept, because it is the same finding this section already records
at 76x and it is the concrete reason W1 is written as a lease on an idle box.

**What closed F2 instead needs no clock.** `decode.audio` declares that it covers
the audio decode, and the audio decode is exactly two calls —
`Ltx2AudioDecoderForward` and `Ltx2VocoderWithBweForward` — each of which F1 gives
a scope of its own. The new case asserts **containment**: each sub-scope's
interval lies inside the interval of the leaf that claims to cover it, the two
together cover at least 90% of it, and the decode and writer leaves never
overlap. Every number comes from one clock in one run and no threshold is
crossed, so contention cannot move the verdict. A **share floor** of 0.05% of the
leaf sum covers `denoise`, `decode.audio` and `decode.video`, the three phases
that carry this render: a name detached from its work measures two function
calls, five orders of magnitude below the floor, while the smallest of the three
holds 3.7%. **That repair covered ONE of those three, and the next section is a
second fresh review demonstrating what the other two could still do.**

**F1 — #1010 named six phases and two of them were folded away.** `decode.audio`
carried `Ltx2AudioDecoderForward` and `Ltx2VocoderWithBweForward` in one leaf,
which on the first artifact was 3.062 s, 25.5% of wall, the second-largest phase
in the table and un-decomposed. The two-stage recipe's latent spatial upsampler
ran inside `phase.prepare`, a leaf whose name does not mention it. Both are now
split as NESTED leaves — `decode.audio.mel`, `decode.audio.vocoder`,
`phase.upsample_latent` — which decomposes them without moving what the table
adds up to, since nested records are excluded from the sum. The vocoder split is
what makes F2's containment invariant expressible at all; the upsampler is gated
by the two-phase DFR case, the only render in `test_ltx2_video` that reaches it.

**F3 — the gate reddened on the next refinement it needed.** The reconciliation in
the SUMS case accumulated every record that was not a `span`, while the emitter's
own `Sum` skips `span || nested`. So the case silently asserted "no nested leaf
has a non-trivial duration" although the header advertises nesting as supported,
and splitting the mel decode out failed it on `CHECK( 0.0118791 < 1e-06 )` — a
message that says nothing about nesting and reads as "the emitter does not
reconcile". Repaired to skip both, which is what made F1's split possible at all.
M6 below re-runs the reviewer's demonstration against the landed split.

**F4 — the artifact had no provenance and sits under `benchmarks/`.** Every
sibling in that directory carries `_source`, `hardware`, `workload`, `headline`
and `footnotes`; the first phase-log artifact carried none of them, so a reader
opening `denoise 8.13 s (67.6%)` beside `decode.audio 3.06 s (25.5%)` in a
directory called `benchmarks/` had nothing telling them the host was contended,
the checkpoint was a two-block fixture, or that the rank of those two phases had
reversed between two runs. That is #1040 and #1087's failure in miniature, inside
the row that exists to stop it. Two changes: **the emitter now writes the caveat
into every phase log it produces** — `notice`, `sum_rule` and `sampler_enabled`,
so the warning travels with the file rather than living in a document a later
reader would have to know to look for — and the committed artifact carries a
`_source` provenance header in the shape its siblings use.

**F5 — `## Now` quoted the superseded run.** It carried 98.33% of a 0.1575 s wall,
which is the run whose phase ranking this section says may not be quoted. It now
carries the artifact's own 99.84%, with both walls named as the non-benchmarks
they are.

**F9 and F10 — the sampler's lifetime.** `StopSampler` hands the thread object out
under `mu` and joins it outside; between those two points an `Open` on another
thread could run `StartSamplerLocked`, which cleared the single `stop` member the
old worker was still reading, so the old worker never exited and the join blocked
forever with two samplers live. The flag is now owned by the worker that reads
it, so nobody else's start can clear it. Separately, nothing but `Reset()` and the
destructor ever stopped the sampler, so a server that rendered one clip kept a
100 ms `/proc/self/statm` read under the process-wide mutex for the rest of its
life and accumulated that idle time into the next table's sample count. The last
`Close` now stops it and `Begin` stops whatever the previous timeline left
running. On this driver that is two starts per process, because the `load` and
`generate` spans each stay open across everything beneath them.

**F6 and F8 — two one-line record defects.** `docs/FEATURES.md`'s ABI capability
table listed the video entry points and omitted `vllm_video_last_phase_log`. And
the comment above the monotone-`start` assertion said the sequence holds *because*
the records are appended in completion order, which is backwards: completion order
is what would break it (`load` closes at start 0.0001 and is appended after
`load.prompt_embeds`, which starts at 0.0625), and what the line actually pins is
`ByStart`'s `stable_sort`.

**A claim this row made about #1126 was too broad, and is narrowed.** See `## Now`
and `## Owed`: the override would wake one named arm, not "a landed residency
policy on every CUDA model".

| # | Mutation | `git diff --stat` | compile | focused gate |
|---|---|---|---|---|
| M4 | leave `decode.video` open across the audio decode and give `decode.audio` its name with no work beneath it — the **attribution** mutation | 1 file, +4/-2 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **26 assertions, 4 failed** — two containment failures (`start >= audio_start`), the mel/vocoder ordering, and the floor at 8.85e-5% against 0.05% |
| M6 | revert F3: reconcile over `span` only, so a nested leaf is summed twice | 1 file, +1/-1 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, 302 assertions, 1 failed, on `fabs((wall - leaves) - unaccounted) < 1e-6` |

**M4 is the acceptance test for F2, and the number that matters is the one
beside it.** Under M4 the pre-existing SUMS case still reports **exit 0, 1 case
passed, 302 assertions, 302 passed**, at 8.00802 s of leaves against 8.01027 s
of wall — 99.97% accounted over 23 entries, every required name present. So the
gate that shipped is blind to M4 exactly as the reviewer said, and the
containment case is what sees it. On the unmutated tree that same case reports
exit 0, 26 assertions, 26 passed, with mel+vocoder covering **99.9994%** of
`decode.audio` and the three floors clearing by 40x (`decode.video`, 1.99%),
286x (`decode.audio`, 14.32%) and 1562x (`denoise`, 78.10%).

**Read those three shares beside the artifact's own and the point makes
itself:** the artifact recorded `denoise` at 67.66%, `decode.audio` at 25.50%
and `decode.video` at 3.73%, and the run above recorded 78.10%, 14.32% and
1.99% on the same binary at the same geometry. The SHARES move by a factor of
two on this box, which is why F2 asserts a floor three orders of magnitude below
them rather than a value, and why no ordering in any table this row produced may
be quoted.

### What a SECOND fresh review found, and what closed it

The repair above asserted containment for `decode.audio`, because `decode.audio`
was the only leaf with sub-scopes. `denoise` and `decode.video` were held by the
0.05% share floor alone, and the non-overlap loop did not mention `denoise` at
all. A second fresh reviewer ran M4's shape one level over, on the phase that
carries this render.

**M7 — the transfer, on the largest phase in the table.** Close `denoise` after
the first sampler step and open `phase.finish` there. No overlap, no nesting, the
sum untouched: 1 file, **+7/-0**, compile rc 0, 0 errors. The containment case
reported **exit 0, 1/1 cases, 26/26 assertions**. The SUMS case reported **exit
0, 1/1 cases, 314/314 assertions**, 99.94% accounted over 24 entries. The table
it emitted put `phase.finish` at **2.129 s, 55.0%** of a 3.873 s leaf sum and
`denoise` at **0.232 s, 6.0%**, on a binary whose honest run measured `denoise`
at **73.4%**. **82% of the denoise was re-labelled and both gates passed.**

That is the same defect M4 named, on the phase #1024 and #1087 are about, in the
table W1 ranks this campaign's levers from. The reviewer recommended a prose
correction. The operator overrode it, and the right call: a prose correction
would have been honest and would still have shipped a gate that cannot see an
82% misattribution.

**What closed it: every carrying phase now has an anchor.** `denoise.step` wraps
the denoiser evaluation inside `Evaluate`, which every sampler arm reaches — the
first-order loop calls it directly and the res_2s loop reaches it through
`hooks.denoise` — so one nested leaf per evaluation says where the sampler
actually spent its time. `decode.video.chunk` runs from the leaf's own open to
the moment the streaming decoder hands a chunk BACK, so its end is a production
event rather than an instrument statement. Both are nested, so
`sum_leaf_seconds` does not move.

**And a third assertion, which sees a transfer directly rather than through a
sum: EXCLUSIVITY.** No other leaf may overlap the window a phase's sub-scopes
span. Containment says "the work is not inside the name"; exclusivity says "a
second name is inside the work". M7 fails both — seven of its eight
`denoise.step` records fall outside the shortened `denoise` leaf, and
`phase.finish` opens in the middle of the denoise window while the steps keep
running around it.

**Why the sub-scope for `decode.video` sat in the driver and not in the VAE.**
The per-chunk decode itself is `AccumulateTemporalGroup` in
`src/vllm/model_executor/models/ltx2_video_vae_tiled.cpp`, one directory outside
this stage's authority at the time. The driver-side anchor is weaker and it is
not nothing: its END is the production callback firing, so a `decode.video` leaf
that closes before its chunk arrives, or that is re-labelled after one, stops
containing the chunk it produced. M10 below is that case. **The VAE-side
sub-scope is no longer owed — the third review's repair placed it**, with
authority extended to that one file for the reason the next section gives.

| # | Mutation | `git diff --stat` | compile | containment case | SUMS case |
|---|---|---|---|---|---|
| **M7** | close `denoise` after the first sampler step and open `phase.finish` there — the **transfer** mutation | 1 file, +7/-0 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **124 assertions, 8 failed** — 7 containment, 1 exclusivity | exit 0, 434/434 |
| M4 | re-anchored onto the repaired tree: leave `decode.video` open across the audio decode and give `decode.audio` a name with no work | 1 file, +2/-2 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **124 assertions, 5 failed** | exit 0, 422/422 |
| M10 | the M7 shape on the VIDEO side: after the first chunk, re-label the decode as the writer | 1 file, +1/-1 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **122 assertions, 1 failed** | exit 0, 422/422 |
| M9 | revert F10: the last `Close` no longer stops the sampler | 1 file, +0/-1 | rc 0, 0 errors | n/a | **RED**, exit 1, 1 case failed, 2 assertions, 1 failed, at 10 samples against 4 |

The SUMS column is not a defect. It is the measurement: **a sum cannot see a
transfer**, in three independent mutations, which is exactly why the containment
case exists and why it is the one that must be run.

On the unmutated tree AT THAT HEAD the containment case reported **exit 0, 1/1
cases, 124/124 assertions**, with `denoise.step` covering **99.667%** of
`denoise` over 8 evaluations, `decode.video.chunk` **99.439%** of `decode.video`
over 2 leaf records, and mel+vocoder **99.995%** of `decode.audio`. Every count
in this section is that head's; the section below re-runs all of them against the
third review's repair and its own numbers supersede these.

**Two more findings from the same review, repaired here.**

*The coverage threshold and the share floor were both mute switches.* At 0.90,
`decode.audio` could have opened 11% early and swallowed 0.13 s — 87% of this
render's entire `decode.video` — while passing everything in the file. Coverage
is a ratio of two intervals measured inside the same contention window, so it is
the one number here that a loaded box does not move, and the audio threshold is
now **0.99**. The three thresholds are deliberately not one number:
`decode.video` is 1.6 ms on this fixture, where a single preempted
`/proc/self/statm` read at a scope boundary is percent-scale, so its threshold is
the loosest at 0.90 and the reason is written beside it. **The 0.05% share floor
is kept and its margin is NAMED rather than tightened**, because the measured
share is precisely the quantity this box destroys: the same binary at the same
geometry has reported `denoise` at 38.1% and at 73.4%, `decode.audio` at 50.9%
and at 16.7%, and `decode.video` at 5.27% and at 2.13%. A floor set at a fraction
of the measured value would be a flake, and under M7 `denoise` sat at 6.0% and
would have cleared any floor a quiet box could justify. The floor now reads "this
name is not detached" and nothing more.

*The emitter baked measured wall times into library source.* The `notice` string
named "0.158 s, 6.138 s and 12.030 s" from one contended box, in `src/`, where no
gate reads it and the next run that refutes it would have to edit a source file
to say so. That is a live number in a place nobody looks — the failure this row
exists to stop, one directory over. The notice is qualitative now and the
artifact's `_caveat` keeps the numbers beside the run they came from.

### What a THIRD fresh review found, and what closed it

Tracked by [#1440](https://github.com/mudler/vllm.cpp/issues/1440), filed and
fixed in the same flow.

The second repair gave every carrying phase an anchor and asserted four things
about it: containment, coverage, exclusivity and non-overlap. **All four are
ratios taken against that anchor.** A third fresh reviewer read that and asked
the one question the previous two had not: what happens when the anchor moves
WITH the leaf. Two mutations, both compiling clean, both exit 0 on both gates.

**M12 — the circularity, on the same phase M7 was about.** Emit `denoise.step`
for `step_index == 0` only, close `denoise` after step 0, and open
`phase.finish` over steps 1-7. The single surviving step scope sits inside the
shortened `denoise` leaf, so containment holds; it covers nearly all of it, so
coverage holds; nothing overlaps and nothing nests. The containment case reported
**exit 0, 1/1 cases, 106/106 assertions** and the SUMS case **exit 0, 350/350**,
over a table putting `phase.finish` at 28.73% of the named leaves against a
`denoise` of 6.11% on a run whose honest denoise was 39.3%. **82% of the largest
phase re-labelled onto its neighbour — M7's defect, on M7's phase, through the
gate M7 produced.**

**M11 — the writer, which had no anchor at all.** Leave the `write(2)`s inside
`decode.video` and emit `artifacts.frames` covering nothing. The leaves stay
disjoint, nothing nests, the sum does not move, exclusivity is blind because
`artifacts.frames` is a DECLARED PARTNER of `decode.video`, and the containment
case reported **exit 0, 124/124** with `artifacts.frames` at **4 microseconds
while nine PPM files were written** (honest: 0.000186 s). The whole writer is
charged to `decode.video`, which is the leaf W5's lever is measured from.

**F11 — the case name claimed more than it checked, and the load had nothing.**
The case was titled "each named phase CONTAINS the work it is named after" and
checked three of about fourteen. Mutation M13 swaps the `load.dit` and
`load.prompt_embeds` scope NAMES — two lines — and every gate in the file stayed
green while 96% of the load's seconds moved onto the wrong name. On the shipped
21B that name holds roughly seven and a half minutes of DiT staging, and it is
the phase W2 and W3 both act on.

**F12 — a recorded justification pointed the wrong way.** The `denoise` coverage
margin was argued as "a percent of a 40 ms leaf", which reads as though the
uncovered part scales with the leaf. It does not: the 16 boundary samples cost
the same wall whether the denoise is 20 ms or 20 minutes, so the RATIO degrades
exactly as the hardware gets faster. The reviewer measured 99.228% on a quiet box
at a 20 ms leaf — an uncovered 0.154 ms — which puts the crossing point at a
denoise leaf of about 3.1 ms. That is a FALSE-RED risk, never a false pass.

**F13 — the artifact's caveat said "five such runs" and listed six walls.**

**What closed M12 and M11 needs no clock, and neither one is a ratio.**

*The RECORD COUNT.* Each anchor must be emitted once per unit of work the RENDER
counted, and the counts come from `Ltx2ConditioningTrace`: `dit_evaluations`,
incremented inside the shared `Evaluate` lambda every sampler arm reaches, and
`video_decode_chunks`, added by this repair and incremented in the driver's own
streaming sink beside `rendered_frames`. Neither is derived from the phase table,
so no placement of a phase scope can move either, and the gate re-derives
`video_decode_chunks` from `Ltx2GroupTilesByTemporalSlice` rather than trusting
it. Under M12 the count reads `1 == 8`.

*NOTHING BUT AN ANCHOR IS NESTED.* A leaf that grows over a NEIGHBOUR does not
overlap it and does not break the sum: `PhaseLog::Open` marks the neighbour
`nested`, which removes it from `sum_leaf_seconds` and from the timeline the
exclusivity check walks. The gate now names the six anchors and refuses any other
nested record.

*And the WRITER got an anchor of its own.* `artifacts.frames.ppm` wraps the
`WriteFileBytes` loop, so it moves with the WRITES rather than with the scope
beside it, and `artifacts.frames` joins the carrying phases with its record count
taken from `video_decode_chunks`. Its coverage threshold is 0.50 and loose on
purpose: the leaf is 0.2 ms on this fixture and carries its own boundaries, so
what binds is the COUNT and the fact that the writer is not `nested`. Measured
97.97% honest. **This sentence used to say "the containment and the count", and
a fourth fresh review showed that is false:** a count of one plus containment
permits the leaf to be TWICE its anchor, because a leaf may grow over adjacent
time nobody named up to its coverage slack. It is harmless on this fixture only
because nothing adjacent to the writer is stealable — `decode.video` is a
declared partner that the `nested` assertion and `CheckWriterIsBesideTheDecode`
both hold — which is a property of the fixture and not of the threshold.

*And `decode.video` got the VAE-side sub-scope this spec recorded as owed.*
`decode.video.vae` opens inside `Ltx2ConvVideoDecodeTiled` around
`AccumulateTemporalGroup`, one record per temporal group, so that leaf finally
has a sub-scope whose ends are both production events rather than statements
adjacent to its own `Open`. Authority for that one file was extended by the
operator for exactly this. **As written at `165db635c` that was a claim about the
SOURCE that no gate held**, and the fourth review below is what closed it: the
anchor was checked for cardinality, for `nested` and for containment in a chunk
window, and its duration was compared against nothing at all.

*And the load got the only check available inside this stage's authority: its
ORDER.* An anchor for `load.dit` would have to open inside
`Ltx2LoadDitFromSafetensors` / `Ltx2StreamDitToDevice` in `ltx2_loader.cpp`,
which this stage does not reach, and the driver has one statement between that
scope's open and its close — so a sub-scope placed there would be the
adjacent-statement anchor M11 and M12 just demonstrated the circularity of. What
is asserted instead is that the load leaves appear in the order the driver runs
them, which a swapped pair of names breaks. **The case title now says what it
checks**, and `### Owed out of W0` below names every leaf whose placement is
still unproven.

| # | Mutation | `git diff --stat` | compile | containment case | SUMS case |
|---|---|---|---|---|---|
| **M12** | emit `denoise.step` for step 0 only, close `denoise` there, open `phase.finish` over steps 1-7 — the **circularity** mutation | 1 file, +9/-1 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **167 assertions, 2 failed**, first on `CHECK( 1 == 8 )` against `dit_evaluations` | exit 0, 374/374 |
| **M11** | the writes stay inside `decode.video` and `artifacts.frames` is emitted covering nothing — the **empty-name** mutation | 1 file, +3/-4 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **194 assertions, 1 failed**, on `artifacts.frames.ppm` enclosed by no `artifacts.frames` leaf, with the leaf reporting 4.008e-06 s | exit 0, 446/446 |
| M11b | the literal reading of the same instruction: both `Close`s below the write, with `artifacts.frames` still around the loop | 1 file, +2/-2 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **126 assertions, 1 failed**, on the writer being emitted `nested` | exit 0, 446/446 |
| **M13** | swap the `load.dit` and `load.prompt_embeds` scope names | 1 file, +2/-2 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **194 assertions, 2 failed**, on the load order | exit 0, 446/446 |
| M7 | re-anchored: close `denoise` after the first sampler step and open `phase.finish` there | 1 file, +5/-0 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **195 assertions, 9 failed** | exit 0, 458/458 |
| M4 | re-anchored: leave `decode.video` open across the audio decode and give `decode.audio` a name with no work | 1 file, +3/-3 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **92 assertions, 1 failed** (a `REQUIRE_FALSE` aborts the case early) | exit 0, 446/446 |
| M10 | re-anchored: after a chunk, re-label the decode as the writer | 1 file, +1/-1 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **195 assertions, 1 failed** | exit 0, 446/446 |

On the unmutated tree at the same head the containment case reports **exit 0,
1/1 cases, 194/194 assertions** and the SUMS case **exit 0, 1/1, 446/446**, with
`denoise.step` covering 97.81% of `denoise` over 8 evaluations,
`decode.video.chunk` 98.78% of `decode.video`, mel+vocoder 99.99% of
`decode.audio`, and `artifacts.frames.ppm` 97.97% of `artifacts.frames`.

**The SUMS column is still not a defect, and it is now the third independent
demonstration of the same thing:** a sum cannot see a transfer, and it cannot see
an anchor that moved with its leaf either.

**The ABI number moved, and that is the one thing in this stage a merge tool
could not have resolved.** `main` took v22 for `vllm_model_params.mmproj_path`
(#821, landed as #1436) while this branch was open, and this branch had written
v22 for `vllm_video_last_phase_log`. Two features under one version is not a
textual conflict: the version is the only question a caller can ask a loaded
library, and one that answers yes for a feature the build does not carry is worse
than none. The phase table is **v23**, and the `>= 22` floor in
`tests/capi/test_capi.cpp`, the block comments in `include/vllm.h`, the
`docs/FEATURES.md` surface row and the `docs/USAGE.md` version line and table
moved with it. The table also gained the v22 row for `mmproj_path` it was
missing.

### What a FOURTH fresh review found

Tracked by [#1446](https://github.com/mudler/vllm.cpp/issues/1446), filed and
fixed in the same flow.

The fourth reviewer called the third repair "by far the strongest iteration" and
re-ran every earlier mutation against it: M12, M11, M11b, M13, M7, M4 and M10 are
all RED, and the record count is the instrument that does it. **Two of the
repair's own stated guarantees still fell to two-line mutations, and both are on
the anchored set the change declares.**

**N1 — sibling anchor names are swappable.** Swap the `decode.audio.mel` and
`decode.audio.vocoder` string literals in the driver (1 file, +2/-2, compile
rc 0). Both gates were green: the containment case **exit 0, 1/1, 194/194** and
the SUMS case **exit 0, 1/1, 446/446**, byte-identical to the honest claim, over
a table reading `decode.audio.vocoder 0.007 s` / `decode.audio.mel 0.220 s` where
the honest table has them reversed. **96.8% of `decode.audio`'s decomposed
seconds land on the wrong name**, on a leaf that is 47.8% of that render's named
leaf seconds and was 25.5% of wall in the first artifact this row shipped, and
[#1010](https://github.com/mudler/vllm.cpp/issues/1010) asks for the vocoder BY
NAME. W1 would have ranked the wrong model.

Every assertion held because **none tied a sub-scope name to a position**: the
counts are `{1, 1}` either way, containment holds for both, `subs` is sorted by
start before the overlap check, and `CheckOnlyAnchorsAreNested` is a
set-membership test. The sharp part is that the third repair added the load-order
assertion (6) *because* M13 showed a sibling name swap moves 96% with everything
green, and then left the identical hole on the one sibling pair inside its own
anchored set.

**N6 — `decode.video.vae`'s placement was not gated at all.** Move the scope off
`AccumulateTemporalGroup` and onto the `buffer.Allocate` in the same loop body
(1 file, +7/-8, compile rc 0): containment **exit 0, 194/194**, SUMS **exit 0,
446/446**. The table then reports `decode.video.vae = 0.000 s` beside a
five-millisecond `decode.video`, and the tile decode — the work the anchor is
named for — sits inside no sub-scope. The gate checked that anchor for
cardinality, `nested` and containment in a chunk window and nothing else;
`Carrying{"decode.video", ...}` lists only `decode.video.chunk` in `parts`, so
the vae contributed **zero** to coverage. The anchor also fired in neither M11
nor M11a, so on the evidence it constrained nothing the gate did not already
hold, and the W5 `## Owed` item it was written to pay was paid in prose only.

Three smaller findings beside them.

**N-count — every decode-side count was demonstrated at N = 1.** The fixture
renders one chunk, so `video_decode_chunks == 1`, `decode.video.chunk == 2`,
`artifacts.frames.ppm == 1`, `decode.video.vae == 1` and the re-derivation
`trace.video_decode_chunks == groups` is `1 == 1`. A count at one cannot
distinguish "once per unit of work" from "once per render", which is exactly the
distinction assertion (0) exists to make; only `denoise.step` was shown at N = 8.
The re-derivation also hardcoded `Ltx2ScaleFactors{8, 32, 32}`,
`Ltx2AutoTileSizeConfig(32, 32, ...)` and `Ltx2CreateTiles(latent_t, 1, 1, ...)`
rather than reading the fixture's own config — safe, because it can only
false-red, but not the independent derivation the pull-request body implied.

**N-render — a per-render counter was compared against a process-wide record
count.** `im.trace` resets on every `Generate` and `PhaseLog` resets only at
load, and `RecordsNamed` did not filter on the `render` field the records already
carry. Correct only because the case rendered exactly once.

**N-slack — a leaf may grow over adjacent UNNAMED time.** The `nested` invariant
sees a leaf swallow a NEIGHBOUR; nothing sees it swallow `unaccounted_seconds`.
The room is each leaf's coverage slack: about 5.3% for `denoise`, 11% for
`decode.video`, 1% for `decode.audio` and up to 100% for `artifacts.frames`. Not
a defect in the repair — it is disclosed here quantitatively rather than closed,
because closing it needs a threshold this box cannot measure.

**What closes N1 and N6.**

*Assertion (1b), THE SIBLING ORDER.* Two sub-scopes of one leaf must appear in
the order `parts` lists them, which is the order the driver runs them, compared
on each name's FIRST record so a part that legitimately repeats
(`decode.video.chunk`) is unaffected and a leaf with one part name is
unconstrained. It is the same shape assertion (6) already uses for the load, and
it proves that a swapped PAIR is a red — not that either name covers the call it
is named after. That needs a scope inside `Ltx2AudioDecoderForward` /
`Ltx2VocoderWithBweForward` and is recorded below beside `load.dit`.

*Assertion (6), THE VAE COVERAGE FLOOR.* `decode.video.vae` must cover at least
half of the render's `decode.video.chunk` seconds. The denominator is the chunk
windows rather than the leaf, because the leaf's last chunk record is the empty
window between the final chunk and the end of the decode. Measured 91.1%, 93.6%,
97.4%, 98.2%, 98.5% and 98.6% over six renders here; the floor is 0.50 because
the uncovered part is the per-group allocation, the overlap blend and the emit —
real work whose share grows as the tile shrinks — plus two instrument boundaries
against a chunk of about a millisecond on this fixture.

*And the counts are now demonstrated at N > 1.* The containment case renders
TWICE: nine frames, which decode in one temporal chunk, and 81, which is the
smallest request this fixture chunks and which the MULTI-CHUNK case above already
derives and asserts. Every per-render assertion runs over each render of one
table, `min_chunks` makes a geometry that stopped chunking a red rather than a
quiet regression to N = 1, and the tiling re-derivation now reads the fixture's
own decoder block list through `Ltx2VideoScaleFactorsFromBlocks` and the size the
render reported, in place of the three constants. The second render is also what
makes the new `render` filter load-bearing rather than hypothetical: that table
holds the load and BOTH renders, so an unfiltered count would be the sum of two.

**One threshold moved, and it is a parameter rather than a loosening.**
`denoise`'s coverage floor is the call's argument now: 0.95 at nine frames and
0.90 at 81. The uncovered part is WORK — the post-process and the Euler or res_2s
step, which no anchor wraps — and its share depends on the geometry. Measured
99.28%, 99.38% and 99.55% at `latent_t = 2` against 96.55%, 96.90% and 97.09% at
`latent_t = 11`, same binary, same box, three runs each. Six points of margin on
the measured value in both cases, which is the rule the other thresholds are set
by.

**Re-run on the FINAL head**, the merge commit `99703ce5a`, and not on the head
each mutation was first measured on. `origin/main` moved TWICE while this branch
was open: `d0eff4f25` (W0-LIVE) edits this row's own driver, and `4712dac40`
narrows `act(gate)` to the input dtype in `src/vt/cpu/cpu_ops.cpp`, which is on
this render's path. A mutation proof inherited from an earlier head is a claim
about a tree that no longer exists, and the whole set was re-applied after each
merge rather than carried forward — N6's count moved from 3 failed to 5 between
the two heads, which is what carrying it forward would have hidden.

| # | Mutation | `git diff --stat` | compile | containment case | SUMS case |
|---|---|---|---|---|---|
| — | honest | (none) | rc 0 | exit 0, 1/1 cases, **554/554** | exit 0, 1/1, **446/446** |
| **R1b** | leave `decode.audio.mel` open across the vocoder call and open `decode.audio.vocoder` EMPTY beside it — the **sibling-boundary** mutation, and the fifth review's finding | 1 file, +2/-2 | rc 0 | **RED**, exit 1, 1 case failed, **554 assertions, 3 failed**, on the per-part floor in each of the three per-render checks | exit 0, 446/446 |
| **R1b mirror** | the same shift the other way: `decode.audio.mel` emptied, `decode.audio.vocoder` covering both calls | 1 file, +2/-2 | rc 0 | **GREEN**, exit 0, 1/1, 554/554 — **OPEN, and recorded above with the measurement that says no floor separates it** | exit 0, 446/446 |
| **N1** | swap the `decode.audio.mel` and `decode.audio.vocoder` scope names — the **sibling-swap** mutation | 1 file, +2/-2 | rc 0 | **RED**, exit 1, 1 case failed, **554 assertions, 6 failed** — three on the sibling order and three on the per-part floor, which is new | exit 0, 446/446 |
| **N6** | move `decode.video.vae` off `AccumulateTemporalGroup` onto `buffer.Allocate` — the **anchor beside the work** mutation | 1 file, +4/-2 | rc 0 | **RED**, exit 1, 1 case failed, **554 assertions, 3 failed**, on the vae coverage floor | exit 0, 446/446 |
| M12 | re-anchored: `denoise.step` emitted for the first evaluation only, `denoise` closed at step 0, `phase.finish` over the rest | 1 file, +8/-1 | rc 0 | **RED**, exit 1, 1 case failed, **479 assertions, 5 failed**, first on `'denoise.step' was emitted 1 time(s)` against `dit_evaluations` = 8 | exit 0, 374/374 |
| M11 | re-anchored: writes stay inside `decode.video`, `artifacts.frames` emitted empty | 1 file, +3/-3 | rc 0 | **RED**, exit 1, 1 case failed, **126 assertions, 1 failed** | exit 0, 446/446 |
| M11b | re-anchored: both `Close`s below the write, writer still around the loop | 1 file, +2/-2 | rc 0 | **RED**, exit 1, 1 case failed, **126 assertions, 1 failed** | exit 0, 446/446 |
| M13 | re-anchored: swap the `load.dit` and `load.prompt_embeds` scope names | 1 file, +2/-2 | rc 0 | **RED**, exit 1, 1 case failed, **554 assertions, 2 failed**, on the load order | exit 0, 446/446 |
| M7 | re-anchored: close `denoise` after the first sampler step and open `phase.finish` there | 1 file, +5/-0 | rc 0 | **RED**, exit 1, 1 case failed, **556 assertions, 26 failed** | exit 0, 458/458 |
| M4 | re-anchored: `decode.video` left open across the audio decode | 1 file, +1/-1 | rc 0 | **RED**, exit 1, 1 case failed, **92 assertions, 2 failed** (a `REQUIRE_FALSE` aborts the case early) | exit 0, 446/446 |
| M10 | re-anchored: after a chunk, re-label the decode as the writer | 1 file, +1/-1 | rc 0 | **RED**, exit 1, 1 case failed, **556 assertions, 4 failed** | exit 0, 446/446 |
| N10 | detach `artifacts.frames.ppm` from the `WriteFileBytes` loop by closing it before the loop runs | 1 file, +1/-1 | rc 0 | **RED**, exit 1, 1 case failed, **554 assertions, 3 failed** | exit 0, 446/446 |

**Twelve mutations, run on BOTH merge heads: `17eba8ce2` and `1160d04b5`, the
head that lands.** The set was re-applied after each merge rather than carried
forward. Across the fourth merge the counts moved and the verdicts did not: N6
reports 3 failed assertions against 5 on `99703ce5a`, and N1 reports 6 against 3
— the last because the per-part floor this change adds catches the name swap as
well as the boundary shift.

**Across the FIFTH merge every count is byte-identical**, all twelve of them,
which is the measurement the merge commit's argument was owed. That merge touches
`src/vt/cpu/cpu_ops.cpp`, the file whose `act(gate)` narrowing moved the counts
last time, and `git diff --numstat` reports 87 insertions and zero deletions
there. "Purely additive" was the prediction; twelve unchanged counts are the
evidence for it, and the prediction is not a substitute for the run.

**SEVEN OF THE TWELVE ARE RECONSTRUCTIONS, and this is the disclosure.** Only
R1b, its mirror, N1, N6 and M13 run from a definition recorded in this tree. The
earlier reviews' own definitions of M12, M11, M11b, M7, M4, M10 and N10 are not
in this tree or on the forge, so what runs under those labels is the mutation
each NAME describes, rebuilt from the row beside it in this table — N10, for
instance, closes `artifacts.frames.ppm` before the `WriteFileBytes` loop instead
of after it, which detaches the writer's anchor from the writes while leaving the
leaf, the count and the nesting intact.

**Two of the rebuilds were wrong on the first attempt, and a bare RED is what
would have hidden it.** A first M4 that never re-closed `decode.video` reddened
the SUMS case, which the recorded M4 does not; a first M12 closed `denoise.step`
early instead of SUPPRESSING it, which still emits the record, and that made M12
indistinguishable from M7 — both 556 assertions with 26 failed, on the same five
failure texts. Both were repaired until the failure SHAPE matched the recorded
one: M4 aborting on a `REQUIRE_FALSE(nested)` after a coverage failure, M12
failing first on `'denoise.step' was emitted 1 time(s)` against
`dit_evaluations` = 8. A reconstruction that merely reds is not evidence that the
recorded mutation reds, and two of seven here proved it.

Every mutation was applied by exact-text replacement that refuses unless the
pattern occurs exactly once, with its `git diff --stat` and its compile return
code printed and the tree restored byte-for-byte and re-verified with
`git status --porcelain` afterwards. A mutation that fails to build, or that
never applies, reads exactly like a passing test.

**The SUMS column is still not a defect,** for the fourth time: a sum cannot see
a transfer, an anchor that moved with its leaf, or a pair of anchor names that
were exchanged.

**The four suites on that same head**, x86 CPU, Release, `-DVLLM_CPP_SERVER=OFF`:
`test_ltx2_video` exit 0, 96/96 cases, 3990/3990 assertions; `test_capi` exit 0,
65/65, 654/654; `test_ltx2_vae` exit 0, 43/43, 3125/3125; `test_ltx2_tiling`
exit 0, 10/10, 915/915.

**One run of `test_ltx2_video` reported three cases THREW and it was not a
verdict.** The box filled to 100% mid-run and the three exceptions read
`No space left on device` and `short write .../frame_000000.ppm`. A full disk
presents as a claim about the code, which is a shape this repository has hit
before with checker refusals; the numbers above are a re-run with 91 GB free.
Check `df` before believing a red.

**No timing conclusion is drawn from this box.** Every threshold above is a
within-run share. The leaf sums moved 0.658 s, 0.706 s and 0.779 s across three
runs of one binary at one geometry while this was written, and earlier runs of
the same case on the same box reported 0.080 s and 0.476 s.

### No claim file, and the checker says why

A `CLAIM-LTX25-RESIDENCY-W0.md` was written and then removed.
`scripts/check-agent-record.py:1712-1714` refuses an active claim whose `Row IDs`
cell holds no ID its `ID_RE` recognises, and that pattern is built from the same
fixed prefix list (`:327-338`) that keeps this campaign out of every matrix —
`LTX25-` is not in it. So a claim file for this stage is a record no checker can
validate, sitting beside a row no matrix carries. The stage's ownership,
evidence and exclusions are recorded here and in the pull-request body instead,
which is where this spec's `### Decisions taken here` already said they would be.

### Owed out of W0

* **The device column has never executed.** See `## Owed`. Every
  `peak_device_bytes` in the artifact is the `-1` no-probe sentinel, and it would
  read `-1` on CUDA too, for the reason #1126 names.
* **A render on the SHIPPED 21.00B checkpoint has not been instrumented**, which
  is W1's whole job. Nothing here says what the table looks like when the DiT
  load is minutes rather than milliseconds; every number in this section is the
  reduced two-block fixture.
* **`GenerateAudioOnly` writes the table but has no gate on it.** The t2a arm
  flushes through the same helper and the case that would check it was not
  written; the video arm is now gated at three levels — the sum, the named
  boundaries, and the containment case.
* **No gate reads a phase table on a host that is not contended,** and this is
  the one W1 must close first. The seconds-differential written for F2 was
  refuted on this box, and the refutation is a property of the box: the same
  binary at the same geometry has moved a factor of 76 in wall and has reversed
  the rank of its two dominant phases. Until W1 takes a lease on an idle box, no
  DURATION in any phase table this row produced may be compared with any other
  duration, including its own from the run before.
* **EVERY LEAF THAT IS NOT ONE OF THE FOUR CARRYING PHASES IS NAMED AND NOT
  ANCHORED**, and this entry used to list only seven of them. The gate anchors
  `denoise`, `decode.video`, `decode.audio` and `artifacts.frames` — one nested
  sub-scope per unit of work the render counted, contained in the leaf, covering
  it, exclusive of other leaves. **Nothing else in the table has an anchor.** The
  full list, so that no later stage inherits an unproven placement by silence:

  | Leaf | State | Who acts on it |
  |---|---|---|
  | `load.dit` | **order only.** Its placement is unproven, and mutation M13 shows what that costs: swapping its name with `load.prompt_embeds` moved 96% of the load's seconds and reddened nothing until the order check landed. An anchor needs a scope inside `Ltx2LoadDitFromSafetensors` / `Ltx2StreamDitToDevice` (`ltx2_loader.cpp`), outside this stage's authority | **W2 and W3, both** |
  | `load.video_vae`, `load.audio_vae`, `load.upsampler`, `load.text_encoder`, `load.prompt_embeds` | order only, same argument | W2 |
  | `conditioning.tower`, `conditioning.connector` | unanchored. These are the 39-100% bound [#1269](https://github.com/mudler/vllm.cpp/issues/1269) is about, so W4 acts on two numbers whose placement no gate holds | **W4** |
  | `generate.setup`, `generate.geometry`, `generate.guiders`, `generate.image_cond`, `generate.audio_input`, `generate.retake`, `generate.trim`, `phase.finish`, `phase.prepare` | unanchored, and each under 0.001% of the leaf sum on the fixture. That is a fact about the DRIVER — they really are bookkeeping — and it is a fact about a two-block fixture, not about 21B | nobody, today |
  | `phase.upsample_latent` | unanchored, and gated only on the reduced two-phase fixture (see below) | W2c |
  | `artifacts.audio` | unanchored; the WAV write, one call | nobody, today |

  The `load` and `generate` records are SPANS and are never summed, so they are
  not in this list; `sum_rule` in every emitted table says which records add up.
* **(1c) DOES NOT HOLD A LEAF RECORD SHORTER THAN 60 ms**, and this
  entry names which ones so no later stage inherits the silence. The span-slack
  bound is per leaf RECORD and its `min` cap binds below
  `2 * kSpanSlackPerRecord`; a record there is reported and not checked, because
  the capped bound is a 50% share and the honest head-plus-tail on those records
  measured 4.6-72.3% of the record itself ([#1559](https://github.com/mudler/vllm.cpp/issues/1559)).

  **That floor is a flat 60 ms, and the shape that would have made it a per-run
  number the case prints was WITHDRAWN** -- see `### The span-slack bound, FIFTH
  and SIXTH shapes`. Measured while that shape existed, over 120 runs on this
  box, it ran 2.4 ms to 838 ms, median 26.6 ms; on this tree the ranges in the
  table below are measured against the flat 60 ms floor and say which records
  are resolvable
  on a QUIET run rather than on every run. A record the run cannot resolve is
  counted in the case's own `N below this instrument's resolution` line.

  | Record | Measured range | What still holds it, and what escapes |
  |---|---|---|
  | `artifacts.frames`, nine-frame render, ONE record | 0.90-6.21 ms | (2) at 0.50 holds it TIGHTLY, because a single-record leaf makes the coverage floor exactly the capped span bound. Plus the (0) count and `CheckWriterIsBesideTheDecode` |
  | `artifacts.frames`, 81-frame render, TWO records | 4.39-61.0 ms | the same names, but (2) is on the SUM, so it is looser per record. (1c) checks a record on the runs where it clears the floor |
  | `decode.video` reopen-after-a-chunk, one or two records | 0.019-5.66 ms | **nothing effective.** A swallow may grow such a record to just under 60 ms -- up to 94x its honest size -- and (2) at 0.90 does not see it: 10% of a 0.17-1.39 s leaf is 17-139 ms of budget, of which 60 ms is 43% at worst and 4.3% at best. Under the third shape the same escape was about 0.5-0.75 ms, so this is roughly an 80x widening on these records |
  | `decode.video` record 0, on a FAST box | 25.0-59.6 ms when it falls | it is normally the third gated record, and it is not reliably one. Over 35 runs it fell below the floor in 4 of 60 shared-box observations and 8 of 30 two-core observations, and in 2 of 20 shared-box runs the whole leaf reported `0 of 2 leaf record(s) checked`. The checked minimum on two cores was 62.6 ms, 2.6 ms above the floor. CI's runner is FASTER than that box, so this gets worse there, not better |

  What escapes is a swallow that leaves the record under the floor, and the row
  above says what that is worth on each class rather than naming an assertion and
  leaving the reader to assume it binds. **The fix is
  not a wider bound, it is naming the time**, which is what #1439 is owed as
  well: a production anchor inside the writer, and one inside the decode's own
  reopen, make those records measurable rather than tolerated. Both are scope
  work in `ltx2_video.cpp` and neither belongs in the repair of a standing red.
* **THE SANITIZER LANES' OWN DISTRIBUTION OF THIS QUANTITY IS NOT MEASURED HERE.**
  The fourth shape retires the second constant, so both sanitizer lanes now
  carry 30 ms where they carried 3 ms. That direction can only turn a red
  green, and the worst sanitizer slack anybody has recorded is 3.354 ms (ASan,
  `artifacts.frames` r2), 8.9x inside it. What is NOT measured is whether a
  sanitizer build's own tail reaches 30 ms on this host, because the disk
  had 7.1 GB free and a sanitizer tree is about 20 GB. Owed by [#1559](https://github.com/mudler/vllm.cpp/issues/1559).
* **AND WHAT IS OWED ABOUT THE ANCHORS THEMSELVES**, which this entry was silent
  about until a fourth fresh review asked. The list above is honest and complete
  for LEAVES; it said the four carrying phases are anchored and stopped there, as
  though "anchored" were a single property. It is three, and only two of them
  hold:

  | About an anchor | State |
  |---|---|
  | It runs once per unit of work the RENDER counted | **PROVEN**, at N = 8 for `denoise.step` and at N = 1 and N = 2 for every decode-side count, over two renders in one table. The counters come from `Ltx2ConditioningTrace` and the chunk count is re-derived from `Ltx2GroupTilesByTemporalSlice` |
  | Sibling anchors under one leaf are in the driver's own order | **PROVEN** for the one sibling pair that exists (`decode.audio.mel` before `decode.audio.vocoder`), and it is an ORDER, not an identity |
  | A sibling anchor does not carry ITS SIBLING'S seconds | **HALF PROVEN**, and the half that is open is named below. A per-part floor holds `decode.audio.vocoder` at 50% of its leaf; nothing holds `decode.audio.mel`, for a measured reason |
  | Each anchor covers the CALL it is named after | **NOT PROVEN, for any of the six.** `denoise.step`, `decode.video.chunk`, `decode.audio.mel`, `decode.audio.vocoder` and `artifacts.frames.ppm` are all opened by a statement ADJACENT to the call. `decode.video.vae` is the only one whose ends are both production events, and what holds its placement is a coverage floor against `decode.video.chunk` — a ratio between two anchors, which is weaker than a count and stronger than the nothing that held it at `165db635c` |

  The last row is the same debt `load.dit` carries and it is not smaller for
  being on the render side. Closing it needs a scope inside the callee —
  `Ltx2AudioDecoderForward`, `Ltx2VocoderWithBweForward`,
  `Ltx2LoadDitFromSafetensors` — which is where the placement stops being a
  statement about where somebody wrote a line.

* **THE SIBLING BOUNDARY MOVES 100% OF A LEAF'S SECONDS BETWEEN TWO NAMES, AND
  ONE OF ITS TWO DIRECTIONS IS STILL OPEN.** This is stated separately from the
  row above because it is a different failure: that row is a PRECISION debt — an
  anchor opened by an adjacent statement measures a few microseconds more than
  its call. This is a whole-attribution TRANSFER, and it is reached without
  touching a scope's name, its count, its containment or its order.

  The shape, which a fifth fresh review named as mutation R1b: leave
  `decode.audio.mel` open across the vocoder call and let
  `decode.audio.vocoder` open and close EMPTY beside it
  (`ltx2_video.cpp:4674-4682`, two lines, identical semantics and identical call
  order). Assertion (1b) passes because the mel still OPENS first, so it proves
  an order and not an identity; the counts are `{1, 1}` either way; containment,
  `nested` and non-overlap all hold; and `min_coverage` is checked on the SUM of
  the parts, so one part covering everything satisfies it. At `1609e1d08` the
  gate returned exit 0, 1/1 and 527/527 assertions — byte-identical to the
  honest tree — while the vocoder, the model
  [#1010](https://github.com/mudler/vllm.cpp/issues/1010) asks for **by name**,
  reported six microseconds of a half-second leaf.

  **The vocoder direction is now closed** by a per-part floor of 0.50 against the
  `decode.audio` leaf (assertion (2b), `Carrying::part_min_coverage`). Honest the
  vocoder measures **99.46% and 94.10%** on the head this lands at (97.85%,
  91.57%, 90.6% and 89.7% on earlier runs) and 97.0% and 99.99% on the 21B
  artifact; under R1b it measures **0.00091% and 0.0035%**. R1b now reds:
  **exit 1, 1 case failed, 554 assertions, 3 failed.** The same floor also
  catches the NAME SWAP, which is why N1 moved from 3 failed assertions to 6:
  under N1 the vocoder name sits on the mel's work and reads 2.76% and 7.68%.

  **The MIRROR direction is NOT closed** — mel emptied, the vocoder scope
  covering both calls — and no floor is available, which is a measurement and
  not an omission. Honest, `decode.audio.mel` holds **0.53% and 5.90%** on this
  head (2.13%, 8.43%, 9.4% and 10.3% on earlier runs) and **0.0004% to 2.9%** on
  the 21B artifact. Under the mirror it holds **0.00063% and 0.0051%**. The
  honest 21B render's 0.0004% is *smaller* than the mirror's 0.0051%, so the two
  distributions overlap and any threshold that reddens the mirror also reddens an
  honest render this row has already produced. The mirror is measured GREEN:
  exit 0, 1/1, 554/554.

  **What W1 inherits, in one sentence, so that it does not inherit it by
  silence:** `decode.audio.mel` may be carrying the vocoder's seconds and no gate
  in this repository would say so. The cost is bounded in one direction — the
  mirror overstates the vocoder by at most the mel's own share, and the vocoder
  is the name #1010 already asks for — and unbounded in the other, which is
  exactly why the vocoder got the floor first. Closing the mirror needs the same
  thing the row above needs and nothing weaker: a scope INSIDE
  `Ltx2AudioDecoderForward`, so that both of the mel anchor's ends are production
  events, in the shape `decode.video.vae` already ships. **Any W1 ranking that
  separates the audio VAE decode from the vocoder must land that scope first, or
  state that it is ranking a pair and not two models.**
* **A LEAF MAY STILL GROW OVER TIME NOBODY NAMED, and the room is measurable.**
  The `nested` invariant sees a leaf swallow a NEIGHBOUR, because the neighbour
  turns `nested` and leaves the timeline. Nothing sees a leaf swallow
  `unaccounted_seconds`. The room is each leaf's coverage slack: **`denoise`
  about 5.3%, `decode.video` about 11%, `decode.audio` about 1%, and
  `artifacts.frames` up to 100%** — that last one meaning the writer's leaf may
  be twice its anchor and pass, which is harmless on this fixture only because
  nothing adjacent to it is stealable. Disclosed rather than closed: a tighter
  floor is a claim about a measured share, and the measured share is the quantity
  this box destroys.
* **F10 IS NOW GATED, and the sentence that said it could not be was wrong.**
  This entry used to read "F10 is an absence — a thread that keeps running —
  which no assertion in this suite is positioned to observe", and asserted that a
  gate would need an injected scheduler. A second fresh review refuted it in
  twelve lines against the PUBLIC surface: `PhaseLog::Samples()` is a counter the
  worker increments every 100 ms, so a worker that outlived its timeline moves a
  number a test can read. Those twelve lines are now
  `ltx2 phase log: the last Close stops the sampler` — the first `PhaseLog`-level
  unit case this instrument has, and the first gate in the file that does not pay
  for a complete render. Reverting F10 (dropping the `TakeSamplerLocked` in the
  last `Close`) makes it fail: **exit 1, 1 case failed, 2 assertions, 1 failed**,
  at 10 samples against 4 over 600 ms of idle. The lesson is the one the row
  keeps re-learning: "no assertion is positioned to observe it" was a claim about
  what had been tried, written as a claim about what is possible.
* **F9 is FIXED AND UNGATED, and that justification stands.** F9 is a race
  between `Reset()` and `Open` whose window is a few instructions wide. It is a
  logic race and not a data race — every access is already under `mu` — so a
  sanitizer would not flag it, and a test that loses it reliably would have to
  instrument the emitter's internals. What the suite proves for F9 is that the
  repair regressed nothing, not that it is detected.
* **`phase.upsample_latent` is gated only on the reduced two-phase fixture.** It
  is the only leaf whose sole reader is the DFR case, because every other render
  in `test_ltx2_video` pins `max_phase = 0`, where the input transform is never
  the spatial upsample.

## W0-live — the lane that runs while the render is alive (#1413)

W0 above landed the table. This section is the half of #1010 that table does not
reach, filed separately as
[#1413](https://github.com/mudler/vllm.cpp/issues/1413) so the two are
attributable to the two changes that made them.

### The gap, in one sentence each

**The table is written by the success path only.** `WritePhaseLog` is reached
after `im.trace.completed = true`, so a render that is killed, aborted by a lease
governor, or still running writes nothing. That is not a corner: #1375 is
`ABORT[92] PROJECTED OVERRUN`, `child exit=-15`, **0 frames**; the spike's rung 1
is `EXIT=137`, 0 frames; rung 2 is `EXIT=1`, 0 frames. The instrument reports on
the runs that finish, and the runs the campaign has are the ones that do not.

**Nothing at all is emitted while the render runs.** `PhaseLog::Open` and
`PhaseLog::Close` record and print nothing, and `VLLM_RENDER_PHASE_LOG_STDERR`
fires inside `WriteJson`, i.e. on the success path again. Between
`ltx2-gen: family=...` and `wrote N frames` a 2.5-hour render is silent, so
**working and hung are byte-identical from outside**, and hours of GPU lease have
been spent telling them apart.

**The unit that costs the wall has no counter.** `denoise` is one leaf. #1375
measures **~162 s per DiT forward** at 21.004 B with **60 forwards structural**
(30 steps x 2 CFG legs — `cfg_scale != 1.0` forces the unconditional branch at
`ltx2_pipeline.cpp:521-523`). ~2.7 h inside one leaf that reports one number, at
the end, if the run survives to write it.

### Why an external sampler is not the fallback

It was tried and it does not work. #1375 records `phase=OTHER` throughout,
because **`eu-stack` unwinds zero frames inside the `rc` worker container**;
`dit_runs=0` is that blind counter and is not evidence of absence. The 162 s is a
wall-clock interval between GPU busy/idle edges, not a profile, and #1375's own
"What is NOT established" names an in-process phase marker as the way to
attribute it. This lane is that marker.

### The shape, and why it is this shape

Two emitters, both on stderr, both on the shipped default.

1. **A boundary line on open and on close.** `+name` when a phase opens,
   `-name` when it closes with its duration and its peak host bytes. The open
   line is the load-bearing one: it means **the last line printed names the phase
   that is currently running**, which is the whole difference between a hang and
   work. A close-only emitter would have printed nothing for the 3002 s
   conditioning stretch, because that stretch never closed.

2. **A tick per DiT forward**, carrying the recipe phase, the sampler step
   `k/N`, the cumulative forward index, elapsed, and `last=` — the seconds since
   the previous tick of the same unit. That `last=` is the per-forward cost
   #1375 could only obtain as an interval between GPU edges, and it is now a
   number the process itself reports, per forward, on every run including the
   ones that die.

   **The tick fires BEFORE the forward, not after.** After is the obvious
   placement and it is wrong here: the line a reader most needs is the one naming
   the forward that is in flight when the run stops. The cost of that choice is
   stated rather than hidden — the last forward of a completed render has no
   `last=` line of its own, and its cost is inside the `-denoise` boundary line.

**Cadence.** 60 tick lines and ~50 boundary lines over 2.7 h. Not chatty enough
to drown a log; not quiet enough that a 162 s forward reads as a hang. That 60
is #1375's config and it moved while this lane was in review:
[#1092](https://github.com/mudler/vllm.cpp/issues/1092) gave
`Ltx2DitForwardDevice` its `perturbations` argument, so the device-resident arm
runs up to four forwards per step rather than two and the same geometry emits
up to 120 ticks. Neither number is printed and neither had to be re-derived:
the forward counter carries no denominator by the paragraph below, and the step
fraction reads this recipe phase's own `sigmas`.

**No denominator on the forward counter, deliberately.** 60 = 30 x 2 is true of
the config #1375 measured and it is not structural in this code: the sampler
decides how many denoiser calls happen and `Ltx2GuidedDenoise` decides how many
forwards each call is (one to four — cond, uncond, ptb, mod). The **step**
counter carries the honest fraction `k/N`, because `sigmas.size() - 1` is exact
and in scope. Printing `forward 24/60` would be an instrument guessing, and this
row exists because guessing is what the campaign has been doing.

### Not behind a flag, and the reason is measured

`VT_H3_PROGRESS` is the existing shape for this in the tree
(`minimax_h3.cpp:776-793`): per-step forward seconds, on stderr, for the
MiniMax-H3 denoise loop. It is **opt-in**, and that is exactly why no LTX-2.5 run
has one — the runs whose profile is needed are the long ones, on a leased box, by
somebody who did not know they would need the number until afterwards. The
failure this closes happened on default settings.

`VLLM_RENDER_PROGRESS=0` silences it, in the same measurement-lane shape
`VLLM_RENDER_PHASE_SAMPLER` and `VLLM_LTX2_POOL_DRAIN` already take here: it
exists so an A/B over what the emitter itself costs runs on ONE binary. It is
never a configuration and nothing in the tree sets it.

### Cost

One `std::fprintf` to stderr per phase boundary and per DiT forward, flushed,
under the process-wide phase mutex the 100 ms sampler also takes — so a tick is a
held global lock plus a flushed write rather than a bare `fprintf`, which at ~110
lines per render is a footnote and is written down rather than left in a profile.
On the 21.004 B geometry that is ~110 writes against 2.7 h of wall. There is **no
per-token and no per-tile emission**: the VAE decode's tile loop and the frame
writer's inner loop are untouched, and the decode's per-chunk boundary lines are
bounded by the chunk count, which is bounded by the frame count.

**`VLLM_RENDER_PROGRESS=0` costs one `getenv` per process, and that took a second
edit to become true.** `ProgressEnabled()` caches the read and `Tick` returns on
its first line, but the `detail` argument is built by the CALLER and was built
unconditionally: a `std::string` from a literal, three `std::to_string`s and four
concatenations, per DiT forward. Against a 162 s forward the magnitude is
nothing; the sentence was still false as written, and a cost claim that is false
in the small is the shape that gets quoted in the large. The call site now asks
`phase::ProgressEnabled()` before it formats anything, which is why that function
is declared in the header instead of living in an anonymous namespace.

### Gate

`ltx2 video: a render through the ABI PRINTS its progress while it runs`, in
`tests/vllm/multimodal/test_ltx2_video.cpp`. It captures **real stderr** with
`dup`/`dup2` across `vllm_video_engine_load` and `vllm_video_generate` — not a
test-only sink — so what it asserts is exactly what a user sees on default
settings, and there is no seam that could be armed in the test and absent in
production. It asserts:

* the boundary lines for `load.dit`, `denoise`, `decode.video` and
  `artifacts.frames` appear, each with a matching `+` and `-`;
* `+denoise` precedes every `dit forward` tick and `-denoise` follows every one
  of them, which is what says the ticks are inside the phase they claim;
* at least two ticks, with **strictly increasing** forward indices — the step
  count moves;
* every tick from the second onward carries `last=`;
* `t=` is non-decreasing over the whole capture.

The strictly-increasing assertion is the non-vacuous half. An emitter that
printed a constant `forward 1` would satisfy "the marker appears" and would
report nothing, which is the failure mode the spike's sampler already had.

### Owed out of W0-live

* **A capture from a real 21.004 B render.** Every number in this section about
  what the lane prints at that scale is derived from #1375's measured 162 s, not
  observed. W1's lease closes it, and the format is fixed here so that run does
  not have to negotiate one.
* **The last forward of a completed render has no `last=` line**, by the
  before-the-forward placement argued above. Its cost is inside `-denoise` and a
  reader who needs it per-forward has n-1 samples, not n.
* **`GenerateAudioOnly` emits boundary lines and no ticks**, because the t2a arm
  reaches the same `Evaluate`. Unverified: no t2a case captures stderr.
* **Nothing writes a partial table on abort.** This lane makes the *phases*
  visible on a killed run; it does not make `phase-log.json` appear for one. A
  signal handler that flushed the table is a separate change with its own
  re-entrancy argument, and it is not this one.

## W5 — the video VAE decode's device arm (#1007)

Issue: [#1007](https://github.com/mudler/vllm.cpp/issues/1007). Branch
`row/LTX25-DEVICE-RESIDENCY-W5`, base `01854663c` (the MERGE BASE with
`origin/main`, verified by `git merge-base`; an earlier draft of this section
said `b537a5344`, which is 8 commits earlier and was never this branch's base —
`AGENTS.md` and [`verification.md`](../verification.md) require the immutable
SHA, so the number is corrected rather than dropped). Kernel-matrix row
`KERNEL-CONV3D`, claim `CLAIM-LTX25-DEVICE-RESIDENCY-W5`.

### W5.0 Why this stage runs before W1, and what it is therefore NOT allowed to claim

`## Now` above orders the stages `W0 -> W1 -> ... -> W5` and says in as many
words that **W4/W5 do not start** if W1 reports `BLOCKED`. W1 is a lease on
`dgx:gpu0`, this stage was dispatched while a render holds that box, and it was
dispatched with the lease explicitly withheld. So the precondition is not met,
and saying otherwise would be the first defect in this section.

The reconciliation is narrow, and it is a scope reconciliation rather than a
waiver. **W1 is a gate on the RANKING, and this stage makes no ranking claim, no
ratio claim and no speed claim at all.** It lands the primitive whose absence is
the reason #1007 is filed — `vt` has no three-dimensional convolution on any
device — plus the dispatch that reaches it. Nothing here asserts that the decode
is the dominant phase, that the 7.25 TFLOP figure is where a render's wall goes,
or that this change makes a render faster. Those are exactly W1's to settle, and
`## Now` stays unamended by this section.

Two things follow, and they are limits rather than caveats:

* **No number in this section is a benchmark.** The only measured quantities are
  a byte comparison and a dispatch count.
* **If W1's table puts `decode.video` low, this change is still correct and
  still reachable — it is just not the lever.** The `vt::Conv3d` op is owed by
  four other host convolution loops (§W5.4), so it does not become dead work
  under either W1 outcome.

### W5.1 Scope

**In.**

1. `vt::OpId::kConv3d`, `vt::Conv3dArgs` and `vt::Conv3d(...)` — the general
   three-dimensional convolution `vt` has never had, on any device.
2. A CPU arm (`src/vt/cpu/cpu_conv3d.cpp`) whose accumulation order is
   **byte-identical** to the host loop the LTX-2.5 goldens were taken through.
3. A CUDA arm (`src/vt/cuda/cuda_conv3d.cu`), one thread per output element,
   walking the same order with `__fmul_rn`/`__fadd_rn` so the two arms are
   bit-identical by construction rather than within a tolerance.
4. The convolution of `Ltx2ConvVideoDecode` — all nine `CausalConv3d` call sites
   in `ltx2_video_vae.cpp`, decoder and encoder — dispatched through that op on
   a queue.
5. That queue threaded from the engine: `Ltx2VideoEngine::Generate` hands the
   decode the queue `Load` resolved, so **placement is decided once at load and
   never per call**, which is upstream's polarity (§W5.2).

**Out, and each named with what owns it.**

* **Residency.** Every buffer between the convolutions — the norms, the SiLU,
  the AdaLN, the noise injection, `DepthToSpaceUpsample`, `AttnBlock3d`,
  `unpatchify` — stays on the host, so a non-CPU queue pays an upload and a
  download per convolution. That is a staged slice; it is named in the commit
  body and the pull request body, and it is owed under `## Owed` with its own
  issue. It is also the shape `parakeet_encoder.cpp:156-169` already ships for
  `vt::Conv2d`: host vectors in, one `vt::` op, host vectors out.
* **A device `Ltx2VaeWeights`.** The stage table calls for one. Without
  residency it would buy nothing but a re-upload of the same weight per call,
  which is what the current shape already does, so it is deferred to the
  residency issue rather than half-built here.
* **[#1011](https://github.com/mudler/vllm.cpp/issues/1011), the memory
  format**, stays a rider and stays owed. `[C, T, H, W]` is what both arms
  consume today.
* **The `AccumulateTemporalGroup` sub-phase** that `## Owed` above owes to W5.
  It is an instrument refinement, not a device arm, and
  [#1439](https://github.com/mudler/vllm.cpp/issues/1439) has the phase table's
  own sum gate RED on `main` at fixture scale — adding leaves under it while its
  ratio assertion is coin-flipping would make two changes indistinguishable.

### W5.2 Upstream chain, re-derived at the pin

Lightricks `LTX-2` @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, the pin
[`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) records. Every line below was
read at that revision while writing this section.

| What | Upstream | What it settles |
|---|---|---|
| the decoder's ONE convolution module | `packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:266` `class CausalConv3d` | the whole decode's arithmetic is this module; `:292-302` builds one `nn.Conv3d` |
| the temporal pad | `convolution.py:305-311` | causal prepends `k_t - 1` copies of frame 0 (`:306-307`); non-causal replicates first AND last `(k_t-1)//2` times (`:309-311`). A `torch.concatenate`, i.e. a MATERIALIZED pad, before the conv |
| the spatial pad | `convolution.py:288-290,299` | `padding=(0, kh//2, kw//2)` with `padding_mode=spatial_padding_mode.value`. torch realises a non-`zeros` `padding_mode` as `F.pad` and then a zero-padded convolution, so upstream materialises this pad too |
| the conv call itself | `convolution.py:312` `x = self.conv(x)` | the single call site the issue names |
| placement is a BUILD-time decision | `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:267-288`, defaulting to CUDA at `:273`; `packages/ltx-core/src/ltx_core/devices.py:29-39` resolves CUDA -> MPS -> CPU | the decoder is never asked per call where to run. It is `.to(device)` once (`:288`) and the latent follows it |
| the pipeline's decode entry | `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1139` | `self._decoder_builder.build(device=self._device, dtype=build_dtype)` — the device is the pipeline's, fixed before the first frame |
| the dtype polarity, and ONLY the dtype | `conv_video_decoder.py:283-286` `weights_dtype = next(self.parameters()).dtype` then `sample = sample.to(weights_dtype)` | the latent takes the WEIGHTS' DTYPE, not the other way round. **NOT a device move:** there is no `.to(device)` anywhere in that function. Placement is settled by the two rows above and by nothing here |

Two secondary oracles, cited for placement only and never for arithmetic, at the
revisions [`oracles/sglang.md`](../oracles/sglang.md) and
[`oracles/vllm-omni.md`](../oracles/vllm-omni.md) pin:

* SGLang `f63458b5b`,
  `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py:71`
  moves the latent to `get_local_torch_device()`.
* vLLM-Omni `a4ea67a21`, `vllm_omni/diffusion/models/interface.py:92` — *"VAE(s)
  (always on GPU)"*.

**What we mirror, precisely:** the polarity, not a call-time switch. The queue is
resolved once in `Ltx2VideoEngine::Load` and the decode runs on it. **What we do
NOT mirror yet:** the tensor staying resident between modules. That is the
residency issue, written down rather than implied.

### W5.3 Design

**The op.**

```
vt::Conv3d(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
           const Tensor* bias, const Conv3dArgs& args)
  out    [Cout, Tout, Hout, Wout]
  x      [Cin,  Tin,  Hin,  Win ]
  weight [Cout * Cin/groups, KT, KH, KW]
  bias   optional rank-1 [Cout]
```

`Conv3dArgs` mirrors `nn.Conv3d`'s constructor keywords 1:1 — `stride_t/h/w`,
`pad_t/h/w`, `dilation_t/h/w`, `groups` — exactly as `Conv2dArgs` mirrors
`nn.Conv2d`'s.

**Two shape decisions, both forced, both stated rather than hidden.**

* **BATCH 1, so `x` is rank 4 and not rank 5.** `vt::Tensor` caps rank at 4
  (`include/vt/tensor.h:12`, `constexpr int kMaxRank = 4`). Raising that cap
  changes the one struct every op in the tree passes, and it is not this stage's
  to make. Batch 1 is what the model decodes — this port drives the LTX-2.5 VAE
  at batch 1 throughout — so the restriction costs nothing here, and it is
  refused by name rather than silently ignored.
* **The weight's two leading axes are MERGED.** torch's parameter is
  `[Cout, Cin/groups, KT, KH, KW]`, rank 5. `Cout` is `out.shape[0]` and
  `Cin/groups` is derivable from `x` and `args.groups`, so the merged
  `[Cout * Cin/groups, KT, KH, KW]` is the same bytes in the same order with no
  information lost — and the wrapper CHECKS `weight.shape[0] == Cout * Cin/groups`
  rather than trusting it.

**The accumulation order, and why it is not `kConv2d`'s.**

`cpu_conv2d.cpp:86-101` keeps ONE f32 accumulator over the whole `(ic, kh, kw)`
sweep. This op must not. `ltx2_video_vae.cpp` documents, with a measured number,
that a single naive f32 sum over all `ci * kernel^3` taps pushes the non-causal
tiled golden to `5.00679e-06` against a `5e-06` tolerance, and that torch's f32
convolution is a blocked GEMM which sums one partial per input channel. So
`kConv3d`'s contract is **one f32 partial per input channel, walked
`(kt, kh, kw)`, added into a running f32 accumulator seeded with the bias**, and
that order is published contract, not implementation detail.

`kConv1d` and `kDepthwiseConv1d` are the precedent for two convolution ops in
this tree deliberately differing in accumulator width and order, with the reason
written at `include/vt/ops.h` `OpId::kConv1d`: *"the ACCUMULATOR WIDTH differs
and is part of the contract"*. `kConv2d` is untouched by this stage for the same
reason its siblings were left alone: changing it would move a shipped model's
numerics.

**Why that matters more than it looks.** It is what makes the routed host arm
**byte-identical** to the unrouted one, so the LTX-2.5 VAE goldens are a real
regression gate on this change rather than a re-baselined one. A tolerance would
have hidden exactly the defects this seam can introduce: a transposed weight
axis, a dropped group offset, a reassociated sweep.

**But the goldens gate the ROUTING, not the ORDER, and that was measured rather
than assumed.** Deleting the dispatch reds 12 of `test_ltx2_vae`'s 44 cases
(§W5.9 M1). Changing only the accumulator ORDER to `kConv2d`'s flat form leaves
`test_ltx2_vae` at **44/44 GREEN** and reds only `test_ops_conv3d` (§W5.9 M3).
The `5.00679e-06` figure `ltx2_video_vae.cpp` records is a mutation that ALSO
narrowed the accumulator width, and reading it as an order-only result is the
mistake this paragraph exists to stop. The op's own order case is therefore not
belt-and-braces — it is the only instrument in the tree that holds the contract,
and the source comments on both sides now say so.

**The padding stays in the model, and that is the mirror.** `CausalConv3d` keeps
building the padded volume — causal and non-causal temporal replication,
`zeros`/`reflect`/`replicate` spatially — and hands `vt::Conv3d` a zero-`pad_*`
call. That is what torch does: `nn.Conv3d` with a non-`zeros` `padding_mode`
runs `F.pad` and then a zero-padded convolution. Pushing LTX's padding enum into
the shared op would put one model's vocabulary in a shared header for no gain.

**The queue.**

`VideoConvSpec` gains a `const vt::Queue* queue` field, so the nine call sites
thread it without nine new parameters. `Ltx2ConvVideoDecode`,
`Ltx2VideoDecode`, `Ltx2ConvVideoDecodeTiled` and `Ltx2VideoDecodeStreaming`
gain a trailing `const vt::Queue* queue = nullptr`. `nullptr` means **the CPU
queue**, not "the old path" — there is exactly one code path, and the device is
a property of the queue. That is deliberate: a `queue != nullptr ? device : host`
ternary would put the interesting branch where nothing in a CPU build can
execute it, which is the shape
[#1426](https://github.com/mudler/vllm.cpp/issues/1426) already records for the
DiT.

On a CPU queue the tensors are **views over the existing host vectors** — no
copy, no extra byte moved, which is what lets the byte-identity claim also be a
no-regression claim for the host arm. On any other device the padded volume, the
weight and the bias are uploaded through `vt::Backend::Alloc`/`Copy` and the
output is downloaded, in the `parakeet_encoder.cpp` `Buf` shape.

**Every device operand outlives the dispatch, and that is not a style choice.**
The first draft scoped the bias buffer to an `if (bias != nullptr)` block, so its
`Free` ran between the kernel LAUNCH and the `Synchronize` inside the download.
On the CPU backend and on the unified-memory fake backend T5 uses, that is
harmless — which is exactly the problem: it is a free of memory a running kernel
is still reading, and **no test on a box without a GPU can see it**. It is
recorded here rather than fixed silently, because the class of defect (a
correctness bug whose only witness is hardware this row was barred from) is the
one a reader of this stage should be most suspicious of.

The production call site is the `decode.video` phase in
`Ltx2VideoEngine::Generate` (`src/vllm/multimodal/ltx2_video.cpp`), which passes
the engine's queue when the engine is on an accelerator.

### W5.4 What else this op is owed by, stated so it is not read as LTX-only

`grep -rni conv3d src include` finds four other host convolution loops with no
device op to route to: `minimax_h3_vae_cnn.cpp`, `minimax_h3_video_vae.cpp`,
`ltx2_upsampler.cpp` and `clip_mmproj_gguf.cpp`. **None of them is rewired by
this stage** — each has its own accumulation order and its own goldens, and
re-baselining four models to land one is a change a reviewer should refuse.
They are named here so the op reads as the shared primitive it is, and so a
later row can find it.

### W5.5 Tests

| # | Case | Where | Red before |
|---|---|---|---|
| T1 | the op's contract and byte-exactness against an INDEPENDENT in-test scalar reference written from the convolution definition in the published order — dtype combinations over f32/f16/bf16, dense/grouped/dilated/strided/padded, ragged extents that fall off every edge, kernel 1 and 3 | `tests/vt/test_ops_conv3d.cpp` | yes — the op resolves no kernel and throws |
| T2 | determinism: thread counts 1/2/4/8 produce `memcmp`-equal output | `tests/vt/test_ops_conv3d.cpp` | yes, with T1 |
| T3 | every committed LTX-2.5 video VAE golden, decoder and encoder, unchanged and not re-baselined | `tests/vllm/models/test_ltx2_vae.cpp` | n/a — the regression gate |
| T4 | the production decode DISPATCHES `kConv3d`, counted through `vt::GetOpProviderStats`, entered at `Ltx2VideoDecodeStreaming` | `tests/vllm/models/test_ltx2_vae.cpp` | yes |
| T5 | a decode on a NON-CPU queue — the fake XPU backend — produces `memcmp`-equal frames to the host arm, and the kXPU provider's selection count moves | `tests/vllm/multimodal/test_diffusion_device_seam.cpp` | yes |

**T5 is the case that makes the device arm more than a compile.** The upload,
the dispatch on a device that is not `kCPU`, and the download all execute on a
box with no GPU, because `FakeXpuBackend`
(`tests/vllm/multimodal/test_diffusion_device_seam.cpp:63-78`) is a
unified-memory backend in a separate executable and `vt::RegisterOp` is a public
API (`include/vt/op_provider.h:127`). What T5 does NOT prove is that a GPU runs
it. That is hardware, and it is owed.

### W5.6 Gates

* Focused: `test_ops_conv3d`, `test_ltx2_vae`, `test_diffusion_device_seam`,
  `test_ltx2_tiling`, `test_ltx2_video`. `test_capi` and `test_ltx2_video`
  contend over `/tmp` fixture directories and are run serially.
* Full: `ctest --test-dir build`, plus `scripts/agent-preflight.sh --staged`.
* **No performance gate and no performance claim.** See W5.0.

**The full gate, on the merged tree, reads 566 of 570.** Clean build, zero
warnings. Every one of this stage's five suites passes — `test_ops_conv3d`,
`test_ltx2_vae`, `test_ltx2_tiling`, `test_diffusion_device_seam`,
`test_ltx2_video`.

**The four failures are `main`'s, and that is BISECTED rather than asserted.**
They are `test_ltx2_text_encoder`, `test_muse_glimmer_text`,
`test_muse_glimmer_text_fallback` and `test_minimax_music3_ar` — text and audio
bf16 tolerance and token-equality gates, none of which this branch touches.
Reverting ONLY `src/vt/cpu/cpu_ops.cpp` to its pre-merge state — the 42-line
gated-activation narrowing that `4712dac40` (`VT-ACT-ROUND-POLARITY`) landed on
`main` — and rebuilding turns all three runnable ones GREEN
(`test_ltx2_text_encoder` 27/27, `test_muse_glimmer_text` 24/24,
`test_minimax_music3_ar` 37/37) with nothing else changed. Restoring the file and
rebuilding brings the red straight back, so the measurement is not a stale
binary. The defect is already filed as
[#1458](https://github.com/mudler/vllm.cpp/issues/1458), which names the same four
suites and the same two exceeded error floors, and `scripts/main-baseline.py`
shows `build-test-cpu` RED on `main`'s own newest scheduled run. Not this row's to
repair: the decision belongs to `VT-ACT-ROUND-POLARITY`.

An earlier pre-merge run also hit `test_nemotron_h_paged_forward`
("No valid attention backend for device type 0 from {FLASH_ATTN: [head_size not
supported]}"), which is [#1371](https://github.com/mudler/vllm.cpp/issues/1371);
merging `main` brought its fix (`9ecaf1bb3`) and it passes post-merge.

### W5.7 Risks

1. **The CUDA arm cannot be compiled on this box.** There is no `nvcc` here and
   CI does not complete in this environment, so the `cuda-fat-build` job's
   verdict is not available before landing. Mitigated by keeping the kernel a
   near-transcription of `src/vt/cuda/cuda_conv1d_general.cu`, which compiles in
   that job today, and by using no construct that file does not. It is a risk,
   not a mitigation: state it in the pull request rather than discover it.
2. **The CUDA arm has never executed.** Nothing in this tree can run it. It is
   registered, it is reachable by construction from the same dispatch T5
   exercises, and it is owed a hardware run.
3. **A host-arm regression would be silent if the accumulation order drifted.**
   T3 is the instrument, and it is only an instrument because the order is
   byte-identical rather than within a band.
4. **The device branch of the production ternary is not executed here**, for the
   reason §12.8 of [`ltx25-guided-video.md`](ltx25-guided-video.md) gives for
   the DiT's: `Ltx2VideoEngine::Load` refuses `device != 0` without a registered
   accelerator. T5 gates the callee on a non-CPU device; the SWITCH is
   [#1426](https://github.com/mudler/vllm.cpp/issues/1426)'s shape, and it is
   named rather than claimed.

### W5.8 Stop conditions

* Stop and report `NEEDS_DECISION` if the routed host arm cannot be made
  byte-identical. Never widen a golden's tolerance to absorb the difference.
* Stop if `vt::Tensor`'s rank cap turns out to be load-bearing for a caller this
  op must serve. Refuse rank 5 by name; do not silently fold an axis.

### W5.9 What was measured, and what stayed green

Base `01854663c` + this branch, x86-64, GCC, CPU-only, `-DVLLM_CPP_BUILD_TESTS=ON`,
Ninja. (The first draft of this line said `b537a5344`; that SHA is 8 commits
before the real merge base and named the wrong tree for every number below. A
fresh review re-ran the focused numbers at the branch head `7f23aadb5` and every
one reproduced, so what was wrong was the ADDRESS and not the measurement.) Every mutation was applied to the source, REBUILT, run, then restored and
REBUILT again with the green reconfirmed — a stale binary reads exactly like a
working mutation.

**A note on this box, because two runs in this record were polluted by it.**
The host hit 100% disk twice while this stage ran, and ENOSPC surfaces in these
suites as a THROWN case with `0 failed` assertions beside `Status: FAILURE!` —
`safetensors: empty file`, `data_offsets end ... exceeds data section size`,
`cannot write .../audio.wav`. Those are I/O, not assertions, and every one of
them passes on retry. `test_ltx2_video`'s final run reads 96 cases, 95 passed,
**3514 of 3514 assertions passed**, with the single failure
*ltx2 keyframe: the first frame is a KEYFRAME that APPENDS* throwing
`cannot write /tmp/.../audio.wav`; re-run alone it is 1 case / 54 assertions
GREEN. Filtering that case by its full name is ALSO a trap and was hit here:
doctest's `-tc` splits on commas, so the name's own comma matches nothing and
prints `SUCCESS!` at exit 0 over `0 cases ran`. The retry above used a wildcard
and asserts the case count moved.

**Red before, green after.**

| | Before | After |
|---|---|---|
| `test_ops_conv3d` | exit 1, `Status: FAILURE!`, 3 of 4 cases THREW `vt: no kernel for op Conv3d (id 123) on device cpu (type 0)` | exit 0, `SUCCESS!`, 4 cases / 2035 assertions |
| `test_ltx2_vae` | 43 cases / 3125 assertions | 44 / 3131, no golden touched |
| `test_diffusion_device_seam` | 6 cases / 43 assertions | 7 / 49 |
| `test_ltx2_tiling` | 10 / 915 | 10 / 915, unchanged |

The `test_ops_conv3d` red was taken with `src/vt/cpu/cpu_conv3d.cpp` removed from
the CMake source list, so the op resolved no kernel. The shape-refusal case
passed in that state, correctly: its `VT_CHECK`s fire in the wrapper before
`GetOp`.

**Mutations.** Each line reports `BUILT`, the compiler error count, that the diff
applied, and the failing assertion BY NAME.

| # | Mutation | BUILT | errors | Result |
|---|---|---|---|---|
| M1 | the `vt::Conv3d` dispatch deleted from `Conv3dThroughSeam` (early return) | YES | 0 | **RED.** `test_ltx2_vae` exit 1, `Status: FAILURE!`, 12 of 44 cases, 21 assertions. `CHECK( stats.selections == 2u )` reads `0 == 2` in *the decode DISPATCHES its convolutions through the vt::Conv3d seam*; both `Conv video decoder matches upstream ltx_core` goldens and both video ENCODER goldens red with it |
| M1' | the same call textually DELETED | **NO** | 2 | Does not build: `unused parameter 'bias' [-Werror=unused-parameter]` and `Conv3dThroughSeam defined but not used [-Werror=unused-function]`. Recorded because a mutation that fails to build reads exactly like a passing test. M1 is the form that BUILDS, and it is the one the verdict rests on |
| M2 | the queue's device ignored; every dispatch forced to `kCPU` | YES | 0 | **RED.** `test_diffusion_device_seam` exit 1, `FAILURE!`, 1 of 7 cases. `CHECK( xpu.selections == 2u )` reads `0 == 2` and `CHECK( cpu.selections == 0u )` reads `2 == 0` in *the video decode RUNS ITS CONVOLUTION on a non-CPU queue, byte-identically*. This is T5's red-before |
| M3 | the CPU kernel's per-input-channel partial replaced by `kConv2d`'s flat accumulator, bias last | YES | 0 | **RED on the op, GREEN on the model.** `test_ops_conv3d` exit 1, 2 of 4 cases, 140 of 2035 assertions; `CHECK( std::memcmp(got.data(), blocked.data(), blocked.size()) == 0 )` reads `6 == 0` in *the per-input-channel partial is NOT the flat accumulator*. `test_ltx2_vae` stays **44/44 GREEN**. Reported as a result, not omitted: it is why the op carries its own order case |
| M4 | the production call site passes `nullptr` instead of the engine's queue | YES | 0 | **GREEN** on `test_ltx2_vae` (44/44) and `test_diffusion_device_seam` (7/7). Green for lack of hardware rather than for lack of a test — see §W5.10. **The end-to-end ABI case is NOT listed here any more.** It was, and a fresh review showed it is not a control for M4: it is equally green under M1, which removes the dispatch entirely. A test that cannot separate the two mutations is evidence about neither — F6 below |

### W5.10 The one link nothing here can gate, stated as a result

`Ltx2VideoEngine::Load` refuses `device != 0` unless the platform seam resolves a
registered accelerator, this build registers none, and no CI job here has a GPU
runner. So `im.on_device` is false in every runnable configuration and the
production ternary already yields `nullptr`: **M4 is a no-op on this box, and it
is GREEN.** That is the same shape §12.8 of
[`ltx25-guided-video.md`](ltx25-guided-video.md) records for the DiT's device
forward, and it is owned by
[#1426](https://github.com/mudler/vllm.cpp/issues/1426).

What that leaves, split honestly into three links rather than one claim:

| Link | What it is | Gated by |
|---|---|---|
| A | `Ltx2VideoDecodeStreaming` -> the decode -> `vt::Conv3d` | the video goldens and the dispatch-count case, both entered at `Ltx2VideoDecodeStreaming`. M1 reds them. That entry is **ONE HOP BELOW** `include/vllm.h`, not the ABI itself: `Ltx2VideoEngine::Generate` calls it (`src/vllm/multimodal/ltx2_video.cpp::Ltx2VideoEngine::Generate`; the anchor is by SYMBOL because both this PR and its review quoted a line number — `:4612` and `:4616` — that the PR's own edits had already moved off the call, which is exactly the rot `ENG-RECORD-ANCHOR-RATCHET` measures). Far stronger than constructing the type by hand, and not the same claim — see F6 below |
| B | a NON-CPU queue -> upload, dispatch, download, byte-identical pixels | T5, on the fake XPU backend. M2 reds it. **W5 BUILT THE INSTRUMENT §12.8 DEFERRED**, and measured what it costs: the callee half of the residency question is now executable on a box with no GPU. See F8 below for what that does and does not settle |
| C | the engine's own `on_device` ternary -> a real accelerator | **nothing here.** #1426, and a GPU |

Link C is one argument in one ternary whose other branch link A gates. Naming it
is the difference between a staged link and dead code.

### W5.11 What a fresh review returned, and what each finding cost

A fresh review of `7f23aadb5` returned **PASS with 8 findings, none blocking the
design**. It resolved every anchor, reproduced every gate number at the true
head, and validated the new device-seam instrument with a mutation of its own.
The repairs are recorded here rather than summarised, because six of the eight
are corrections to THIS document and a correction that does not say what it
corrected is indistinguishable from a rewrite.

**F1 — the merge, and the value that was neither side's.** The branch went behind
`origin/main` while it was in review, and `check-agent-record.py` conflicted.
`main` bumped its `KERNEL` count 52 -> **53** for `KERNEL-DFLASH2-GROUPED-CONV`
(#1314); this row bumped 52 -> **53** for `KERNEL-CONV3D`. **Both sides said 53
and the merged value is 54**, so a resolution that picks a side is wrong
whichever side it picks. `.agents/kernel-matrix.md` union-merges to 54 rows, so
53 would have RED the gate rather than passing quietly — it fails safe — but it
was resolved to 54 deliberately and both justification paragraphs were kept.
`.agents/engine-matrix.md` conflicted on `ENG-RECORD-ANCHOR-RATCHET`, where this
row had bumped five line anchors by +8 and `main` had replaced the same five with
SYMBOL anchors for that precise reason; `main`'s side was taken whole.
`ltx2_video_vae_tiled.cpp` conflicted a third time, after `fdefb4529`
(`LTX25-RESIDENCY-W0`) landed a `decode.video.vae` phase scope around the call
this row had given a `queue` argument — resolved by keeping BOTH, which no
automatic merge would have produced.

The op-enum auto-merge was re-verified rather than trusted: extracting `OpId`
from the merge base, from `origin/main` and from the merged tree gives 124, 125
and 126 names, `main` adds `kDFlashGroupedConv` alone, the merged tree adds
`kConv3d` alone, and `kConv3d` is still the last name before `kCount`.

**F2 — a citation used for more than it says.** `conv_video_decoder.py:283-286`
was cited for *"the latent following the weights"* inside paragraphs about DEVICE
PLACEMENT. At that anchor upstream reads `weights_dtype =
next(self.parameters()).dtype` and then `sample = sample.to(weights_dtype)`: a
DTYPE follow, with no `.to(device)` anywhere in that function. The substantive
placement claim was already carried correctly by `single_gpu_model_builder.py:273`
and `blocks.py:1139`, so nothing downstream was wrong and no conclusion moved —
what changed is that the anchor no longer stands behind a claim it cannot
support, in `include/vllm/model_executor/models/ltx2_video_vae.h`, in
`src/vllm/multimodal/ltx2_video.cpp`, in the `## Owed` row for
[#1451](https://github.com/mudler/vllm.cpp/issues/1451) above, and in the
polarity table in §W5.8.

*The two ranges are reconciled, and the reconciliation is stated rather than
guessed.* This tree cites the same dtype block two ways: `:282-284` (in the stage
table above, and in [`ltx25-decode-speed.md`](ltx25-decode-speed.md) for
`output_dtype = sample.dtype`) and `:283-286` (for `sample =
sample.to(weights_dtype)`). They OVERLAP and name adjacent statements of one
block rather than contradicting each other, so the stage table now reads
`:282-286` and says `DTYPE only`. **LIMITATION, recorded because it bounds this
repair:** the Lightricks/LTX-2 checkout is not present on this host, so the exact
bytes at those lines were NOT re-read here. The correction is conservative in the
only direction that matters — it NARROWS a claim onto anchors that two
independent citations in this tree already support — and re-reading upstream is
owed to whoever next has that checkout.

*The index row could not be corrected in place, and this paragraph is the
correction.* [#1451](https://github.com/mudler/vllm.cpp/issues/1451)'s row in
[`issue-index.md`](../completed/issue-index.md) carries the same overreach. That file is
append-only and carries `merge=union`, so editing the row means a REMOVED line —
which `check-issue-index-append-only.py` refuses, and which union-merge would
DUPLICATE rather than merge. Appending a second #1451 row would be worse than the
defect it repairs. So the row stands and the correction lives here, which is what
`AGENTS.md` §Records asks for.

**F3 — the gate evidence named the wrong base SHA.** §W5 and §W5.9 said base
`b537a5344`. The true merge base is `01854663c`, eight commits later. The review
re-ran every focused number at the real head and all of them reproduced, so the
measurements were sound and only their ADDRESS was wrong;
[`verification.md`](../verification.md) requires the immutable SHA, so both
stamps are corrected in place rather than deleted.

**F4 — one public document was missing a limit the commit body claimed it
carried.** The head commit body asserts that STATUS, BENCHMARKS, FEATURES and
USAGE *"all say the same two limits"*. Three of them named both #1451 and #1452;
`docs/BENCHMARKS.md` named only #1452. The sentence was made TRUE rather than
softened: BENCHMARKS now carries the host-loop limit too. Under a squash the
false sentence would not have landed, but `AGENTS.md` also permits a local
operator merge, under which it would have.

**F5 — index rows were not at the tail, and MEASURING it reversed the answer.**
#1451 and #1452 sat five rows above the end of `issue-index.md`, against
`AGENTS.md`'s *"Append a row at the end."* It is gate-invisible: the append-only
checker inspects REMOVED lines only and has no positional check.

The obvious reasoning says leave it — moving a row means removing a line, which
is the one edit that checker refuses. **That reasoning is wrong here, and a probe
said so.** `check-issue-index-append-only.py` diffs the MERGE BASE against the
head, not the previous commit, and these two rows do not exist at the merge base:
this row added them. Relative to `01854663c` a move within their own addition
removes NOTHING. Measured on a detached scratch commit built through a private
`GIT_INDEX_FILE` — because a working-tree mutation of a commit-reading checker
returns 0 without ever reading the mutated bytes, which is #1417's lesson — the
moved index gives `OK: issue index append-only`, rc 0. So the rows were moved to
the tail and the written rule is satisfied rather than excused.

The union-merge hazard the rule exists to prevent is untouched by this, for the
same reason: a line that is new on this branch cannot be an EDIT of a line
another branch also has.

**F6 — the only test that enters at the ABI is PIXEL-BLIND.** Under M1, with the
whole convolution dispatch removed so the decode emits all-zero pixels, the
end-to-end case *"ltx2 video: an ABI client loads, detects and generates through
vllm.h"* stays **GREEN** — the same case count and assertion count as the
unmutated baseline, with the filter confirmed to have matched.

This is NOT a "nothing lands dead" failure and it is not recorded as one. The
code is unambiguously reached: `Ltx2VideoEngine::Generate` calls
`Ltx2VideoDecodeStreaming`, the video goldens and the dispatch-count case enter
there, and M1 reds twelve of their forty-four cases. Entering one hop below the
ABI is far stronger than constructing the type by hand. What the finding changes
is how link A above should be READ — `Ltx2VideoDecodeStreaming` is one hop below
`include/vllm.h`, not the entry point — and it removes the ABI case from M4's
evidence row, because a case that is equally green under M1 and under M4
discriminates neither and was never a control for either.

**F7 — the shape contract diverged from the contract it advertises, and it is
FIXED rather than documented.** `vt::Conv3d`'s output-extent formula used C++
integer division, which truncates toward zero; torch FLOORS. The two differ only
when `(in + 2*pad - dilation*(k-1) - 1)` is negative AND stride > 1. At
`tin = 2, k = 3, stride = 2, pad = 0` the numerator is -1: torch computes
`floor(-1/2) + 1 = 0` and raises *"Output size is too small"*, while truncation
computed `-1/2 + 1 = 1` and ACCEPTED a `Tout` of 1, convolving over taps the
stride had skipped.

**Unreachable from LTX**, which is why no model gate here could hold it:
`CausalConv3d` materialises a pad of at least the kernel on every axis, so the
padded extent is never below the kernel and the numerator is never negative. The
review reasoned it from source without executing it. It was reproduced BY
EXECUTION here, in a three-step cycle over one build directory:

| Step | BUILT | compile err | `git diff --stat` | `test_ops_conv3d` |
|---|---|---|---|---|
| the floored guard in place | YES | 0 | `src/vt/ops.cpp` 54 +++ | exit 0, **4 of 4 cases, 2036 of 2036 assertions**, `Status: SUCCESS!` |
| M-F7: reverted to C++ truncating division | YES | 0 | `src/vt/ops.cpp` 43 +++ | **exit 1, 1 of 4 cases failed, 1 of 2036 assertions.** `CHECK_THROWS( vt::Conv3d(q, to, tx2, tw, nullptr, strided) ) did NOT throw at all!` at `tests/vt/test_ops_conv3d.cpp:446` |
| restored byte-for-byte | YES | 0 | `src/vt/ops.cpp` 54 +++ | exit 0, 4 of 4, 2036 of 2036, `SUCCESS!` |

Every rewritten file was `touch`ed twice around a sleep before each rebuild,
because a file that keeps its old mtime makes ninja SKIP the rebuild and the
no-op then reports `BUILT=YES compile_err=0` over the PREVIOUS binary. The
`git diff --stat` column is the other half of that trap: a mutation that never
applied reads exactly like a passing test, and the 54 / 43 / 54 line counts are
what prove it moved and moved back.

**The first attempt at this case was a FALSE red, and it is recorded because it
nearly landed.** It reused the enclosing fixture's `tx`, whose `Tin` is 3. At
`Tin = 3` the span is `3 + 0 - 2 - 1` = **0**, which is non-negative, so floor and
truncation AGREE and torch accepts that geometry too — the `CHECK_THROWS` was
failing against CORRECT behaviour. It was red before the fix and still red after
it, and only that second symptom exposed it. A case that is red for the wrong
reason is indistinguishable from one that is red for the right reason, right up
until the fix fails to move it.

*The decision, and why it went this way rather than into a header note.* §W5.4
offers this op to other models as a SHARED SEAM, and the header at `vt::Conv3d`
states that it mirrors `nn.Conv3d`'s shape contract. A shared seam that silently
disagrees with the contract it advertises is the more expensive option later, and
the fix is a strict no-op on every reachable path: for a non-negative numerator
floor and truncation are the same value, so no shipped geometry, golden or
tolerance moves. Documenting the divergence instead would have cost every reader
of the seam the exact guarantee the header sells. The sibling SUBCASE at stride 1
is kept and now says why it cannot substitute: at stride 1 the two divisions
AGREE, so that case is blind to this defect.

*The sibling ops carry the same expression, and they are FILED rather than
fixed.* `vt::Conv2d` (`src/vt/ops.cpp:2749-2750`) and `vt::DepthwiseConv1d`
(`:2883`) compute their extents the same truncating way.
[#1471](https://github.com/mudler/vllm.cpp/issues/1471) owns them and is listed
under `## Owed` above. They are not repaired here because that means two more
red-first cases and a fresh review over ops this row does not own, and because no
caller is yet known to reach a negative span on either — an audit, not an
assumption. `Conv1dOutLength` (`:2886`) and `ConvTranspose1dOutLength` already
refuse a negative span explicitly, so three of the five conv wrappers in this
file were right all along, which is the strongest argument that the other two are
a slip rather than a decision.

**F8 — one sentence overstated, and the true version is stronger.** §W5.10 said
link B is *"what W5 adds that §12.8 could not"*. §12.8 of
[`ltx25-guided-video.md`](ltx25-guided-video.md) actually says the branch is
*"deferred on cost, not closed on impossibility"*, and its link B is the **DiT**
forward `Ltx2DitForwardDevice` — a different path from this row's VAE
convolution. **W5 does not close
[#1426](https://github.com/mudler/vllm.cpp/issues/1426).**

What W5 did do is BUILD the instrument §12.8 deferred, and measure its cost:
**129 lines added to `tests/vllm/multimodal/test_diffusion_device_seam.cpp`**, an
executable that already existed (`d415c931d`). That is the number that matters to
#1426, because #1426's blocking assessment was a COST assessment, and a cost
measured at ~130 lines inside an existing binary makes the same technique cheap
to apply to the DiT. Weakening a deferral is not closing it, and this row claims
only the first.

**The non-blocking suggestion was taken, and moved one level up.** The review
offered a one-line log on the first CUDA `kConv3d` dispatch naming #1452, to turn
a silent first execution of never-compiled, never-run code into an announced one.
It is in `vt::Conv3d` in `src/vt/ops.cpp` and fires once per process on the first
NON-CPU dispatch of any device type, rather than inside the `.cu` file. Two
reasons: the same argument holds for every accelerator arm this seam gains, and
host code is code this box COMPILES and `test_diffusion_device_seam` EXECUTES —
putting the answer to "never-compiled code runs unannounced" inside
never-compiled code would have been the defect wearing the repair. It does not
reduce the residual risk the review named, which is real and unchanged: a
compiling-but-wrong CUDA kernel would produce wrong pixels on `--device cuda`,
and no gate anywhere in this tree catches it. That is
[#1452](https://github.com/mudler/vllm.cpp/issues/1452).

### W5.12 The gate after the review repair, re-stamped

The numbers in §W5.9 were taken at `7f23aadb5` over base `01854663c`. This is the
rerun after the review repair and after `origin/main` was merged TWICE, because
it moved under the branch while the repair was in flight.

| | |
|---|---|
| base (merge base) | `01854663c` |
| `origin/main` merged in | `fdefb4529`, then `7481a2eec` |
| build | `cmake -DCMAKE_BUILD_TYPE=Release -G Ninja`, gcc, x86-64, CPU-only, `-DVLLM_CPP_BUILD_TESTS=ON` |
| full build | `ninja -C build` **exit 0, 0 compile errors**, 586 targets |
| `scripts/agent-preflight.sh` | **exit 0, `All gates green.`**, and 0 SKIPPED |
| `ctest -j 6` | **574 of 574 passed, 0 failed**, exit 0, 130 s wall; 3 skipped by their own guards (`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`, `test_qwen35_paged_engine`) |

Focused, each run on its own:

| Suite | Cases | Assertions | Status |
|---|---|---|---|
| `test_ops_conv3d` | 4 of 4 | 2036 of 2036 | `SUCCESS!`, exit 0 |
| `test_ltx2_vae` | 44 of 44 | 3131 of 3131 | `SUCCESS!`, exit 0 |
| `test_ltx2_video` | 101 of 101 | 4109 of 4109 | `SUCCESS!`, exit 0 |
| `test_capi` | 65 of 65 | 654 of 654 | `SUCCESS!`, exit 0 |
| `test_diffusion_device_seam` | 7 of 7 | 49 of 49 | `SUCCESS!`, exit 0 |
| `test_ops_conv2d` | 4 of 4 | 1631 of 1631 | `SUCCESS!`, exit 0 |

`test_ops_conv2d` is in that list deliberately, and it is a CONTROL rather than a
gate for this row: F7 changed a shape expression that `vt::Conv2d` shares in form
and does not share in code, and #1471 owns the sibling. Its 4 of 4 says the
Conv3d repair did not reach it, which is the claim, not that `vt::Conv2d` is
correct.

**Two SKIPs became two runs, and that is the point of re-stamping.** Before the
merge, `agent-preflight.sh` SKIPPED `commit-trailers` and `commit-style` with
*"Neither gate reported anything about this tree"*, because the branch was behind
`origin/main`. A skip is not a pass, so §W5.9's claimed `exit 0` was not
reproducible for those two gates at the time it was written. Both RUN and PASS
here, against `origin/main` `7481a2eec`.

**No environmental red had to be discounted, which was not expected.** `main`
carried two live reds when this repair started: #1439, and #1458, which reddened
`test_ltx2_text_encoder`, `test_muse_glimmer_text`,
`test_muse_glimmer_text_fallback` and `test_minimax_music3_ar`. #1458's fix
landed in `5ba99ac4e` and is inside the second merge, so all four are green here
and are counted in the 574 rather than excused. `test_engine_core_proc`,
`test_async_llm` and `test_cpu_x86_llamacpp_floor`, which starve or drift under
parallel `ctest`, also passed at `-j 6` on this run and needed no serial rerun.

## The phase-coverage residue bound (#1494, and what it measured about #1439)

`ltx2 video: the three carrying phases contain their work and the load keeps its
order` was RED on `main` in the `build-test-cpu` lane, on the `denoise` leaf, on
both of the fixture's renders. This section records the repair and, more usefully,
the measurement that says which repair was available.

### The residue is TWO things and the gate charged them to one number

A probe over the interval structure of the `denoise` leaf, both renders, this
tree:

| render | leaf | head | tail | interior gaps |
|---|---|---|---|---|
| 9 frames | 8.0-45.3 ms | 3.2-15.9 us | 19.6-49.1 us | 7 x 62-118 us |
| 81 frames | 39.3-72.1 ms | 4.8-15.9 us | 53.5-102.5 us | 7 x 463-549 us |

So about 5% of the un-covered seconds are the leaf's own boundaries and about 95%
are the SEVEN gaps between consecutive `denoise.step` records. Those gaps are the
sampler's per-step update: real work, once per evaluation, scaling with the
latent, and NOT the "16 boundary samples of fixed cost" the previous comment at
the site reasoned from. That reasoning put the crossing point at a 3.1 ms leaf
and left the floor at 0.95; `main` then reported the red at leaves of 13.7 ms and
27.5 ms, four to nine times above it.

### The share is a property of the box, so no floor near it is stable

`denoise` coverage on an UNCHANGED tree, nine-frame arm: 99.55%, 99.38%, 99.28%,
99.228% (the row's own box); 98.84%, 98.77%, 98.52%, 98.23%, 94.14% (a second x86
box as its load moved); 96.85%, 94.60%, 94.60% (the box #1494 measured); 92.39%
and 88.85% (the GitHub runner, from CI jobs `96506970274` and `96466616360`). The
81-frame arm spans 97.09% down to 85.85% over the same set. A 0.95 floor is below
roughly half of its own honest distribution. The polarity is the one #1439
recorded and it is the tell: a short serial stretch between two long parallel ones
is what loses to a contended scheduler, so the ratio moves with the box and not
with the tree.

### The repair

Assertion (1c), the SPAN SLACK, is new and is what now carries this leaf beside
the (0) record count. It bounds the leaf's seconds lying outside the span its
sub-scopes occupy -- computed per leaf record, because `decode.video` has two or
three -- at a FLAT 0.25 ms. That is the only place a swallowed phase can land, it
is two instrument boundaries rather than N intervals of work, and it measured
11.4-64.3 us across all four leaves and both renders.

The bound was first written `max(1 ms, 2% of the leaf)` and a fresh review broke
that shape twice, both correctly:

* **The share never tightens.** Inside a `max` it is pure slack, and it is
  LARGEST where the leaf is largest -- under load `denoise` reached 3.82 s with
  38 us of slack against a 76 ms bound, and `decode.audio` 3.07 s with 39 us
  against 61 ms, 500x to 2200x. At production scale a swallowed phase of several
  seconds would have passed, which is the opposite of what a share was added for.
  The quantity does not grow with the render -- every measurement on both boxes
  is at or under 135 us whatever the leaf did -- so a constant holds on all of
  them and is strictly tighter at every leaf size. `kSpanSlackShare` is gone.
* **The 1 ms term made the CHECK VACUOUS on the smallest leaf.** `span_slack <=
  leaf_seconds` holds by construction, so any bound at or above the leaf cannot
  fail. `artifacts.frames` measures 0.826-0.983 ms here, under the 1 ms bound:
  forcing 100% slack on every leaf reddened only ten of twelve checks, and that
  leaf's two PASSED AT 100% SLACK. It was also a coin flip, because on slower runs
  the same leaf measured 1.35-13.9 ms and did bite -- so whether the assertion
  existed depended on box speed, the exact property this change removes.

Both are closed. The constant is 0.25 ms: ~2.4x the worst value measured here
(102.5 us) and ~1.9x the worst the review measured (135 us), a multiple of a
measurement rather than a number fitted to today's run, and below the smallest
leaf so the check can fail. And a `REQUIRE(span_bound < leaf_seconds)` now sits
above it, so a leaf that ever shrinks beneath the instrument's noise floor reds
and says so instead of passing quietly. Re-run of the same forced-strict probe:
**twelve of twelve red, `artifacts.frames` included.**

The coverage floor (2) stays, and `denoise`'s parameter moves from 0.95/0.90 to
0.75 on both arms: 10.85 points below the worst honest observation, kept for the
range where honest and defective separate widely, which is the argument the 0.50
floors on `decode.video.vae` and `decode.audio.vocoder` are already set by. The
geometry dependence the previous comment parameterised is real and is SMALLER than
the box dependence, so one number below both arms replaces two numbers sitting
inside each. **Nothing was deleted:** a parameter moved and an assertion was added
beside it.

No claim is made that (1c) is tighter than 0.95 IN NUMBERS, because it is not:
5% of a 13.7 ms leaf is 685 us against this 250 us, and against the 1 ms the bound
first carried it was 1.5x LOOSER. The point is a different one. The old bound
covered head, tail AND interior together while the interior alone measured 1046 us
on the runner that reported it, so the budget was spent before any swallow, no
honest tree could satisfy it, and it therefore bounded nothing at all. The new one
is spent by 11.4-64.3 us and does not move with the box.

### Evidence

Two mutations in `src/vllm/multimodal/ltx2_video.cpp`, each applied by exact-text
replacement refusing unless the pattern occurs once, compile status printed, tree
restored and `sha256sum`-verified afterwards.

| # | Mutation | compile | result |
|---|---|---|---|
| honest | none | rc 0 | exit 0, 1/1 cases, **566/566** assertions, 4 consecutive runs at loadavg 6.7-10.2, and 3 more at loadavg 54-61 |
| **M1** | `denoise` opens at the top of `phase.prepare` instead of after it, so the prepare's seconds become un-named time inside the leaf | rc 0 | span slack moves **14 us -> 139 us** and PASSES; a fresh review reproduced it independently at **26.6 us -> 124.6 us**, 554 assertions green. `phase.prepare` is only ~125 us on this fixture, under even the tightened 0.25 ms bound. The coverage floor (2) misses it too, moving 0.4 points, so this is not a regression against the shape that shipped -- but it IS the instrument's resolution at this geometry, and it is now stated AT THE SITE rather than only here |
| **M1 + calibration** | M1 with the bound lowered to 0.1 ms, to exercise the comparison itself | rc 0 | **RED**, exit 1, 1 case failed, 566 assertions / **3 failed**, each on (1c)'s own message quoting `0.000100121s` against `0.0001s`. The assertion is live and compares the quantity it names |
| **M2** | `denoise` opens right after `generate.setup` opens, swallowing the whole conditioning and preparation stretch | rc 0 | **RED**, exit 1, 496 assertions / **14 failed**, on (3c) `CheckOnlyAnchorsAreNested` -- the swallowed leaves become `nested`, which is the assertion that owns that shape |

Full suite on the landing head: **102 cases, 101 passed, 4194 assertions, 4193
passed**. The one failure is #1439's `CHECK(leaves >= 0.95 * wall)`, which this
change deliberately does not touch; see below. It flaps by box load on this host
and is GREEN in CI, so a run of this suite here reports either 101 or 102 passed
and neither is a statement about this change.

### The span-slack bound, third shape: it is the BUILD that scales it (#1531 lane)

The flat 0.25 ms constant passed the plain lane and reddened BOTH sanitizer lanes
the moment #1532 made them able to run. The premise behind it -- "this quantity
does not scale" -- was right about RENDER SIZE and wrong about BUILD
CONFIGURATION, and every measurement behind it had come from uninstrumented
builds. The slack is the cost of opening and closing a phase scope, and a
sanitizer instruments exactly that path: ASan checks a shadow byte on every access
in it, TSan additionally keeps per-access happens-before state under a lock. It is
a different machine, not a noisier one.

| lane | that test's wall | worst span slack, CI | worst span slack, 20-core box |
|---|---|---|---|
| plain | 259 s | 121 us | 74.7 us |
| `address,undefined` | 1165 s | 730 us | 461 us |
| `thread` | 2116 s | 1658 us | 895 us |

The ordering is the build and not the box: both columns rank the same way, and CI
is worse throughout because those runners are two-core and contended.

**A NORMALISED bound would be better, and there is no normaliser.** The old 0.95
coverage floor tolerated instrumentation ACCIDENTALLY, by being a ratio of two
quantities that inflate together -- TSan makes both `covered` and `leaf_seconds`
8x bigger and the ratio survives. The span slack is ABSOLUTE, so the overhead
lands on it undiluted. That is the real reason the "does not grow with the render"
premise did not generalise: it does not grow with the RENDER, it grows with the
INSTRUMENT.

Five candidate normalisers were tested over 24 observations (four leaves x two
renders x three build configurations), scored by the spread of the resulting
ratio, worst over best:

| candidate | spread | undefined |
|---|---|---|
| `slack / smallest sub-scope` | 78630x | 0 |
| `slack / leaf_seconds` | 4598x | 0 |
| `slack / mean sub-scope` | 2377x | 0 |
| `slack / mean interior gap` | 418x | 3 of 24 |
| **`slack` itself** | **44.6x** | 0 |

**Every ratio is worse than the raw quantity**, the two most obvious by two orders
of magnitude, and the interior-gap normaliser is additionally undefined for a leaf
with a single sub-scope, which `artifacts.frames` is. Conditioned on the BUILD
instead, the raw quantity is tight: 20.1-93.6 us plain, 128-569 us under ASan,
52.1-895 us under TSan, i.e. about 4.7x, 4.4x and 17.2x. Configuration is the
variable that explains this quantity, so a per-configuration constant is the most
stable bound available here rather than a fallback taken for want of anything
better.

**A single constant is impossible on this fixture, and the numbers say so
exactly.** The worst slack is 1.658 ms while the SMALLEST leaf is 2.77 ms
(`artifacts.frames`, ASan, measured). Those are 1.67x apart, so no number is both
comfortably above the noise and below the leaf. This was not reasoned out in
advance -- a 3 ms constant was tried and it reddened the anti-vacuity
`REQUIRE(span_bound < leaf_seconds)` on exactly that leaf.

**The bound is therefore three things, and the third one is the repair:**

    span_bound = min(kSpanSlackPerRecord * leaf_records, 0.5 * leaf_seconds)

* **Per leaf record**, because `decode.video` has two or three and each carries
  its own pair of boundaries. This is a count of instrument boundaries, never a
  share, so it is strictly tighter on multi-record leaves.
* **Configuration-aware**: 0.25 ms plain, 3 ms under either sanitizer, selected at
  compile time. ~2.1x the worst plain measurement and ~1.8x the worst sanitizer
  measurement. The sanitizer margin is the thinner and is disclosed rather than
  papered over. The guard is nested `#if`s, not one `||` expression, because GCC
  does not define `__has_feature` and must still PARSE the call: the one-line form
  compiled on the TSan leg only because `||` short-circuited before reaching it,
  which is a green that means nothing.
* **Capped at half the leaf with `min`, never `max`.** This is the polarity the
  earlier review removed, inverted. Under `max` the share was a FLOOR, so it only
  ever loosened the bound and loosened it most where the leaf was biggest -- 2200x
  on a multi-second leaf. Under `min` it is a CEILING, so it only ever tightens:
  on a big leaf the constant binds and production-scale behaviour is exactly the
  constant, and on a leaf small enough that the constant would swallow it, the
  leaf binds instead. It also makes `span_bound < leaf_seconds` true BY
  CONSTRUCTION, so the anti-vacuity REQUIRE becomes an invariant on the FORMULA
  rather than a live risk, and the forced-strict probe is guaranteed to red on
  every leaf rather than happening to.

**Verified in all three configurations**, four runs each, on the landing head:

| lane | result | worst slack | smallest leaf | sanitizer findings |
|---|---|---|---|---|
| plain | 4/4 green, 578/578 | 74.7 us | 0.887 ms | - |
| `address,undefined` | 4/4 green, 578/578 | 418 us | 2.92 ms | 0 ASan, 0 UBSan |
| `thread` | 4/4 green, 578/578 | 870 us | 7.02 ms | 0 TSan |

Forced-strict probe re-run under BOTH plain and ASan: **twelve of twelve red** in
each. Full `test_ltx2_video` on the plain build: 102 cases, 101 passed, 4194
assertions, 4193 passed, the single failure being #1439's untouched
`CHECK(leaves >= 0.95 * wall)`.

### #1439 is NOT repaired here, and the measurement says why not

The obvious move was to give the sibling assertion the same two-term bound. The
probe refuses it. Instrumenting where that residue sits, on a render with
`wall = 0.343348 s` and `unaccounted = 0.0123761 s`:

* the first leaf record starts at `0.0113197 s`, so **11.3 ms -- 91% of the whole
  residue -- is ONE contiguous un-named interval at the head of the `load` span**,
  before `load.dit` opens;
* the interior gaps between named leaves total about 0.92 ms;
* the tail past the last leaf is about 0.12 ms.

That is not instrument noise. It is a real, nameable startup phase: the ~164 lines
between `phase::Scope load_span("load")` and `phase::Scope dit_phase("load.dit")`
-- device resolution, the platform probe, file discovery and option parsing. An
absolute slack big enough to stop this assertion flapping would have to be larger
than 12 ms, and would therefore HIDE that phase. That is the mute switch this
repair exists to avoid, so the assertion is left exactly as it is.

**What #1439 is owed is what its own text asks for first: name the time.** A
`load.open` scope over that prologue drops the residue to about 1 ms of 343 ms,
0.3%, and makes the 0.95 floor honest instead of marginal. It is a production
scope in `ltx2_video.cpp` and it owes the phase names published in
`docs/models/ltx-2-5.md`, so it is a unit of work rather than a rider on a red
repair. Note the polarity while it is open: the GitHub runner measured 95.52% and
97.69% on the two `main` runs that reported the `denoise` red, so this assertion
is GREEN in CI with 0.52 and 2.69 points of margin, and it is the local boxes that
see it fail.

### The span-slack bound, fourth shape: it was inside its own distribution, and its multiplier came from the artifact (#1559)

The third shape landed in `6b48edb2c` and a fresh review measured it on an
UNMUTATED tree at that head. **It reds.** Not on a mutation, not under a
sanitizer, not on a geometry nobody runs: on the plain `build-test-cpu` lane the
change was written to keep green.

That is the second time this bound was calibrated on too few runs, so the
calibration is the part of this section worth reading first.

#### The population, and what it took to see the defect

| batch | runs | box | loadavg | runs RED at `6b48edb2c` |
|---|---:|---|---|---:|
| A | 33 | as shared, 20 cores | 79.7-118.0 | **3** |
| B | 10 | as shared + an eight-way spin load | 91.7-145.9 | **2** |
| C | 10 | pinned to two cores, `taskset -c 18,19` | 84.5-105.6 | 0 |

53 runs, 424 leaf checks, 636 leaf-record observations, one binary, one build
directory, `CMAKE_BUILD_TYPE` empty exactly as `build-test-cpu` configures it.
**Five runs red, on an unmutated tree.**

Two things about that population are worth carrying forward. The first is that
batch C -- two cores, the geometry CI actually has -- is the FASTEST and the
CLEANEST of the three: `denoise` runs 0.64-2.7 s there against 14-41 s in batch
A, because a 20-thread pool on a box at loadavg 100 is worse than two threads on
two cores. An oversubscribed many-core box is the harsh case for this quantity,
not a small one. The second is that no configuration reproduces the tail on
demand: the 10.032 ms observation is one of 636, and the eight-way spin load that
was supposed to provoke it produced 0.963 ms.

Three runs of a green suite is what the third shape was accepted on. Five of 53
runs red, so a run reds about one time in ten and three runs had about a
one-in-four chance of showing it. They did not, and the bound landed.

#### F1 -- the constant was inside the honest distribution of the quantity it bounds

`CHECK(span_slack <= span_bound)` compares two instrument boundaries against a
constant. The constant was 0.25 ms on the plain lane. **The honest quantity
reaches 10.032 ms on the same lane, on the same binary, with nothing mutated**,
and the five reds below say nothing whatever about the tree:

| run | leaf | records | slack | bound | leaf record |
|---|---|---:|---:|---:|---|
| A13 | `decode.video` r2 | 3 | 1.161 ms | 0.75 ms | 1.257 s |
| A16 | `artifacts.frames` r2 | 2 | 1.387 ms | 0.50 ms | 4.826 ms |
| A18 | `decode.audio` r1 | 1 | **10.032 ms** | 0.25 ms | 6.783 s |
| B6 | `denoise` r2 | 1 | 0.410 ms | 0.25 ms | 24.181 s |
| B8 | `denoise` r2 | 1 | 0.963 ms | 0.25 ms | 24.655 s |

**AND RECALIBRATION ALONE CANNOT FIX IT, which is the finding under the
finding.** The bound is capped at `0.5 * leaf_seconds`, so on a short leaf record
no constant can lift it. `artifacts.frames` r2 record 1 measured 1.361 ms of
slack on a 4.826 ms record: the cap forbids any bound above 2.413 ms, and the
honest value is 56% of that. On the shortest records the cap forbids any bound at
all worth having -- `decode.video`'s reopen-after-the-last-chunk record is 19-89
us of which 7-32 us is its own two boundaries, so the capped bound of half the
record sits at 35-72% of a distribution whose honest values reach 72%.

So the cap, which the third shape added to keep the check able to fail, is itself
the thing that makes a short record ungateable. That is not a defect in the cap.
It is the instrument telling the truth: below a certain record length this
quantity is the instrument and not the tree.

#### F2 -- the multiplier was read from the artifact under test

The bound was `min(kSpanSlackPerRecord * leaves.size(), 0.5 * leaf_seconds)`.
`leaves.size()` is whatever `RecordsNamed(table, c.leaf, c.render)` returned,
guarded only by `REQUIRE(!leaves.empty())` and tied to no independent number, so
**a defect that emits one extra leaf record enlarges its own budget by a whole
constant**. Assertion (0) counts the SUB-SCOPE records against
`Ltx2ConditioningTrace`; nothing counts the leaf's own.

The slack was also SUMMED, so a swallow concentrated in one record of a
three-record leaf drew on the whole `3 * kSpanSlackPerRecord`. The comment at the
site claimed the opposite -- that summing "makes the bound STRICTER on
multi-record leaves rather than looser" -- and that is false for exactly this
reason. It was load-bearing rather than academic: under ASan `artifacts.frames`
r2 measured 3.354 ms of slack against a 6 ms bound and passed only because that
leaf has two records.

#### F3 -- the failure message named the wrong cause

The message ended: *"unlike the coverage share below, this quantity is two
instrument boundaries and does not move with the box"*. It is the first sentence
the next person reads on a red, and it tells them to look for a swallowed phase.

Measured on ONE build configuration, ONE host, ONE binary, the same leaf record
across runs:

| leaf record | best | worst | spread |
|---|---:|---:|---:|
| `decode.audio` r1 rec 0 | 25.0 us | 10032.0 us | **401.8x** |
| `decode.video` r2 rec 0 | 7.7 us | 1114.9 us | 144.3x |
| `artifacts.frames` r2 rec 1 | 14.6 us | 1360.9 us | 93.3x |
| `denoise` r2 rec 0 | 73.2 us | 963.3 us | 13.2x |
| `decode.video` r1 rec 0 | 6.8 us | 88.4 us | 13.1x |

The half of the claim that survives is "does not grow with the RENDER", and that
half is still measured and still true: the 81-frame arm's slack is not
systematically larger than the nine-frame arm's, on leaves that differ by an
order of magnitude in seconds. The half that does not survive is "does not move
with the box". The same two comments that carry the stale claim also still quote
"a flat 0.25 ms", which had not been true since the third shape made the bound
per-record, configuration-dependent and capped. All three are corrected.

#### The fourth shape

    for each leaf record r:
        span_bound = min(kSpanSlackPerRecord, 0.5 * record_seconds)
        REQUIRE(span_bound < record_seconds)          # the formula tripwire
        if span_bound < kSpanSlackPerRecord:          # the cap bound: unresolvable
            report, do not check
        else:
            CHECK(slack(r) <= span_bound)             # == kSpanSlackPerRecord

Four changes, and each answers one finding.

* **PER RECORD, NOT PER SUM.** `leaves.size()` is gone from the arithmetic
  entirely, so no count the artifact chose enters the bound. The claim the
  message makes -- that ONE record's head and tail are bounded -- is now the
  claim the code tests, and it is strictly tighter than the sum on every leaf
  with more than one record.
* **THE `min` CAP IS KEPT, AND IT IS PER RECORD.** The inversion is the reason:
  `min(K, 0.5 * r) <= K`, so the share is a CEILING and can only tighten, exactly
  as the third shape argued. It has to be per record rather than per leaf sum,
  because a cap on the sum does not make the forced-strict probe red: with three
  equal records the worst record's slack is a third of the leaf and a half-leaf
  cap would not bite. Per record it is guaranteed -- `span_bound <= 0.5 * r < r`,
  and forced-strict drives the slack to exactly `r`.
* **WHERE THE CAP BINDS, THE RECORD IS BELOW THIS INSTRUMENT'S RESOLUTION AND IS
  REPORTED RATHER THAN CHECKED.** Two reasons, and the second is the measured
  one. A capped bound is no longer an absolute quantity; it is "head plus tail is
  at most half of this record", which for a single-record leaf is exactly what
  `covered >= 0.5 * leaf_seconds` already asserts -- and a new
  `REQUIRE(c.min_coverage >= 0.5)` holds every case here to that floor, so the
  implication is executable rather than asserted. And on the records where the
  cap binds, the honest head-plus-tail measured 4.6% to 72.3% of the record, so a
  50% share is INSIDE its own distribution. That is the 0.95-coverage-floor
  defect again, and gating on it would be the mute switch this whole assertion
  replaced.
* **ONE CONSTANT, 30 ms, FOR EVERY BUILD CONFIGURATION.** That is **2.99x the
  worst of 636 leaf-record observations**, the same idiom as the 2.1x and 1.8x
  the third shape used,
  and something has to bound it from ABOVE, because the floor `2 * K` must stay
  under the shortest record worth gating.
  **What bounds it is `denoise` and `decode.audio`, and the reason first written
  here was falsified by a bigger batch.** This paragraph said the two-core
  geometry keeps `decode.video` record 0 gated at 0.106-0.54 s. Measured over 35
  runs of the repaired shape it is not reliably gated at all: it fell below the
  60 ms floor in 4 of 60 shared-box observations (25.0-26.6 ms) and 8 of 30
  two-core observations (52.4-59.6 ms), the checked minimum on two cores was
  62.6 ms, and in 2 of 20 shared-box runs the leaf reported `0 of 2 leaf
  record(s) checked`. CI's runner is faster than that box, so it straddles there
  too. The conclusion stands on the other two leaves instead, which is where it
  should have stood: `denoise` and `decode.audio` are gated in 35 of 35 runs, are
  seconds long in every configuration measured, and hold 96%+ of the leaf-seconds
  -- a floor of 200 ms would still gate them, and a floor of 2 s would not. The
  first claim came from the 53-run batch's `decode.video` range and was not
  re-derived when the repaired shape was measured; it is corrected here rather
  than quietly dropped. The third shape's two constants --
  0.25 ms plain, 3 ms under either sanitizer -- collapse to this one, because the
  plain lane's own scheduler tail is now 3x the largest sanitizer slack anybody
  has recorded (3.354 ms, ASan, `artifacts.frames` r2). A second constant would
  have to clear the same tail, so it would read 30 ms against 33 ms. Both
  sanitizer lanes therefore get a LOOSER bound than they carried, which can only
  turn a red green.

**The skip cannot hide a swallow, and the direction is the reason.** A swallowed
phase makes a leaf record LONGER, never shorter, so it moves a record TOWARD the
gated set and never out of it. What escapes is bounded and stated: a swallow that
leaves the record under 60 ms in total.

#### What it can no longer see

Stated so the next reader does not have to derive it.

1. **Any swallow under 30 ms, on every leaf.** The previous number was
   0.25 ms and it was not real, because it reddened 5 of 53 honest runs.
   Mutation M1 -- `denoise` opened over `phase.prepare`, about 125 us -- was
   already invisible at 0.25 ms and stays invisible. What changes is the size of
   an invisible swallow, from 0.25 ms to 30 ms.
2. **Everything about a leaf record under 60 ms.** On this fixture,
   measured over 53 runs:

   | record | measured range | what escapes, stated as a ceiling |
   |---|---|---|
   | `artifacts.frames`, 9-frame render, 1 record | 0.90-6.21 ms | little: ONE record means (2) at 0.50 IS the capped span bound, so head plus tail stays under half the leaf. The (0) count and `CheckWriterIsBesideTheDecode` also hold |
   | `artifacts.frames`, 81-frame render, 2 records | 4.39-61.0 ms | more, because (2) is on the SUM of two records rather than on each. (1c) checks a record on the runs where it clears the floor |
   | `decode.video` reopen-after-a-chunk, 1-2 records | 0.019-5.66 ms | **anything up to the 60 ms floor**, i.e. up to 94x the record's honest size. (2) at 0.90 is not a backstop here: 10% of a 0.17-1.39 s leaf is 17-139 ms, so 60 ms is 4.3-43% of a budget the honest records barely touch. The third shape's escape on these records was about 0.5-0.75 ms |
   | `decode.video` record 0 | 25.0-59.6 ms when it falls below | the same ceiling, on the runs where it falls. It is gated on most runs and not on all: 4 of 60 shared-box and 8 of 30 two-core observations were below the floor, checked minimum 62.6 ms |

   **THE HAND-OFF TO (2) IS ONLY TIGHT FOR A SINGLE-RECORD LEAF**, and that is
   one of the four rows above. `covered >= min_coverage * leaf_seconds` is a
   statement about the leaf SUM. On a one-record leaf it is exactly the capped
   span bound; on a multi-record leaf the budget is shared, and on
   `decode.video` the leaf is three orders of magnitude longer than its reopen
   records, so the coverage floor has room those records could never fill.
   `REQUIRE(c.min_coverage >= 0.5)` at the site guards the single-record case and
   claims nothing about the others.

   On the two-core geometry CI actually runs -- measured with `taskset -c 18,19`,
   where the render is *faster* because a 20-thread pool on a contended box is
   worse than two threads on two cores (`denoise` 0.198-3.33 s pinned against
   2.10-11.29 s unpinned) -- the reliably gated set is `denoise` and
   `decode.audio` alone.
3. **The interior gaps, unchanged.** (1c) never saw them and still does not; (2)
   is what bounds them, and #1439 is what is owed about the un-named time.

The fix for 2 is the same fix #1439 is owed and is not a wider bound: **name the
time**. An anchor inside the writer, and one inside the decode's own reopen,
make those records measurable instead of tolerated. That is production scope work
in `ltx2_video.cpp` with its own row, and it is recorded under
`### Owed out of W0`.

#### Evidence

Everything below is one x86_64 host, one build directory, `CMAKE_BUILD_TYPE`
empty as `build-test-cpu` has it, GCC. Every mutation was applied by exact-text
replacement that refuses unless the pattern occurs once, the compile status is
printed, and the tree was restored and `sha256sum`-verified afterwards
(`b7a9e14065aea4dd483d8bf0f2357dab85efceb9b554efab8a2a3cfa501f16d3`).

**Red before.** 53 runs at `6b48edb2c`, unmutated, five red. Population and
values in the table above. This is the falsification, and it is the one the
third shape did not have.

**Green after.** 40 runs of the fourth shape, same binary, same case:

| batch | runs | box | loadavg | result | records checked | worst checked slack |
|---|---:|---|---|---|---:|---:|
| PA | 20 | as shared | 60.2-100.5 | 0 failures | 179 | 3.409 ms, 8.8x inside |
| PB | 10 | + an eight-way spin load | 61.4-101.5 | 0 failures | 90 | 0.183 ms |
| PC | 10 | `taskset -c 18,19` | 81.2-88.2 | 0 failures | 78 | 0.222 ms |

347 records checked, 333 reported below the instrument's resolution, zero
failures. `denoise`, `decode.audio` and `decode.video` record 0 are gated in
every configuration; `artifacts.frames` cleared the floor once in 20 PA runs and
never in PB or PC, which is the straddle `## Owed` records.

**Mutation 1, forced-strict.** `span_slack` forced to the whole record, so every
leaf is 100% un-anchored. Compile rc 0. Result: exit 1, **9 of 9 checked records
RED**, 592 assertions / 9 failed, at 9.96 s, 0.32 s, 2.89 s, 11.28 s, 0.92 s and
5.62 s against the 30 ms bound. No checked record survives, which is what the
per-record `min` cap buys and is now guaranteed rather than observed.

**Mutation 2, the concentrated swallow -- and this is F2's proof.** 45 ms, i.e.
`1.5 * kSpanSlackPerRecord`, added to the slack of leaf record 0 only. Both
shapes compiled at the IDENTICAL constant of 30 ms, so only the formula differs.

| shape | compile | result | `decode.video` |
|---|---|---|---|
| the SUM, `leaves.size()` multiplier | rc 0 | exit 1, 578 assertions / 9 failed | **PASSES**: 45.0 ms of slack against a bound of 0.06 s (r1, 2 records) and 0.09 s (r2, 3 records) |
| per record | rc 0 | exit 1, 592 assertions / 9 failed | **REDS**, at 45.0 ms against 30 ms |

The old formula let one record of a multi-record leaf spend the whole leaf's
budget, and the leaf with the most records was the one it protected most. That
is the opposite of what the comment at the site claimed.

**What is NOT measured here.** No sanitizer run. The disk had 7.1 GB free and a
sanitizer tree is about 20 GB, so the sanitizer lanes' own distribution of this
quantity stays owed and is recorded under `## Owed`. The direction is safe --
they move from a 3 ms bound to 30 ms, which can only turn a red green -- and the
worst sanitizer slack anybody has recorded, 3.354 ms, is 8.9x inside the new
bound. The per-record split is the part that could newly red there, and it is
that same 3.354 ms observation that shows it does not.

### #1439 repaired: the prologue is now a NAMED phase, `load.open`

The section above left #1439 with one route standing and refused the other. This
is that route taken. **The issue stays OPEN**, for the reason the last two
paragraphs of this section give: the repair is measured, the regime where the
floor bites is not. **Nothing about the assertion changed.**
`CHECK(leaves >= 0.95 * wall)` reads byte-for-byte as it did; the residue it
divides by is smaller because a phase that was un-named is now named.

**What moved, in three lines of production code.**
`src/vllm/multimodal/ltx2_video.cpp` opens `phase::Scope open_phase("load.open")`
immediately after the `load` span and calls `open_phase.Close()` immediately
before the `load.dit` block. `Scope::Close` rather than a nested block is the
shape `render_phase_log.h` documents for this driver, whose regions are
sequential statements in one function; a `Fail()` throw out of the prologue
closes it through the destructor instead, which costs nothing because a load that
throws writes no table.

**What the phase contains** is what the probe named: device resolution, the
platform probe, the two refusals, the device-byte probe installation, the DiT
safetensors header open (`SafetensorsFile::Open`) and the LoRA option parsing.

#### Red first, and the red is the production scope's absence

Measured on the shared x86_64 dev box, 20 cores, `Release`,
`VLLM_CPP_CUDA=OFF`, one build directory. The test change was applied and built
ALONE, so the red arm is this tree with the gate asking for the name and no
production scope to answer it:

| arm | `ltx2 video: a render through the ABI emits a phase table that SUMS to wall` |
|---|---|
| test change only | **RED**, exit 1, 447 assertions / 1 failed: `the phase table names no 'load.open' phase` |
| test + production scope | **GREEN**, 459 assertions / 0 failed |

That red IS the reachability mutation this change owes: deleting the production
call site is precisely the first arm, and the focused gate does not stay green
without it.

**A second mutation pins the CLOSE, not only the OPEN.** Sinking
`open_phase.Close()` to below the `load.dit` block — so the prologue scope
swallows the DiT load — compiles, keeps the SUM case GREEN at 459/459, and is
caught by the #1440 nested-record rule instead: `the three carrying phases
contain their work and the load keeps its order` goes RED, 595 assertions /
1 failed, with `'load.dit' is emitted NESTED, and it is not one of this render's
anchors`. Tree restored and md5-verified between arms
(`src/vllm/multimodal/ltx2_video.cpp` identical before and after).

#### What the residue did, and the honest limit on this number

Six runs per arm of the case alone, same binary per arm, box under other
sessions' builds at loadavg 24-31:

| arm | `unaccounted_seconds` (ms) | median | wall (s) |
|---|---|---:|---|
| before | 2.190, 3.907, 2.783, 2.602, 3.174, 3.042 | **2.91** | 7.18-9.24 |
| after | 0.950, 4.706, 1.221, 1.206, 0.815, 0.936 | **0.94** | 8.15-9.89 |

`load.open` is emitted as a LEAF at `t=0.000` (not `nested`, not a span) and
measures about 1.3 ms here. It is the same interval, and the drop in the residue
is its duration.

**Two things this run does NOT establish, said plainly.** First, it does not
reproduce the red: at loadavg 24-31 the render's wall is 7-10 s rather than the
0.22-0.58 s of #1439's population, and that is the exact regime #1439 identified
as the FLATTERING one — a slower render passes, because the residue grows more
slowly than the wall it is divided by. Both arms sit near 99.96%, so this box
measures the residue in SECONDS, which is the scale-free quantity, and not the
ratio. Second, it does not reproduce the **11.3 ms** head the earlier probe
recorded; the prologue is about 1.3 ms at this build type and with the fixture
warm in the page cache. The direction and the mechanism are measured here. The
magnitude at which the floor bites is inherited from that probe and is not
re-measured, so the claim "this makes the 0.95 floor honest instead of marginal"
rests on the earlier measurement plus this one, and a quiet-box run at fixture
scale stays owed.

**Not owed: a `docs/models/ltx-2-5.md` edit.** The section above says this change
"owes the phase names published in `docs/models/ltx-2-5.md`". It does not, and
the reason is that the document publishes none: `## Inspect a render` describes
`sum_leaf_seconds`, `unaccounted_seconds` and the two environment switches, and
names no individual phase. Checked rather than assumed, because a sentence
written when a document had a different shape is how a stale obligation survives.

### The span-slack bound, fifth shape: the instrument measures its own resolution (#1572, #1576)

> **THIS SHAPE DOES NOT LAND. Read `### The span-slack bound, FIFTH and SIXTH
> shapes` below before anything here.** A fresh review broke this estimator by
> execution, a sixth shape built to repair it reddened an unmutated tree, and
> both are withdrawn: (1c) on `main` still carries the fourth shape's 30 ms
> constant. Everything measured in this section was measured; nothing concluded
> in it about an issue's state survived. The section is kept whole rather than
> edited, because its measurements are what the withdrawal rests on.

Four issues describe one behaviour -- `test_ltx2_video` decides by box load --
and every repair so far has replaced one constant with a larger constant.

| Issue | What it recorded | State after this change |
|---|---|---|
| [#1576](https://github.com/mudler/vllm.cpp/issues/1576) | (1c) is a fixed wall-clock budget with an allowance for INSTRUMENTATION and none for LOAD, measured swinging 171x on one binary | **STILL OPEN** -- this shape was withdrawn |
| [#1572](https://github.com/mudler/vllm.cpp/issues/1572) | (1c) redding intermittently on `main`, `decode.video` at 0.00256913 against 0.00075 | **STILL OPEN** -- this shape was withdrawn |
| [#1536](https://github.com/mudler/vllm.cpp/issues/1536) | the phase-coverage floors missed by 0.18-0.8% on an idle box, while the suites that ARE load-dependent passed | **STILL OPEN.** Its thesis is contradicted by the 120-run population below, which was taken on an unmutated tree and stands; but "refuted" was written as a consequence of this shape landing, and it did not land |
| [#1470](https://github.com/mudler/vllm.cpp/issues/1470) | one case of 96 false-redded under load, four greens after, case identity never captured | **identity and rate measured, STAYS OPEN**, see below |

The history of the number they are about is the argument for not choosing
another one: 0.95 of the leaf, then 0.25 ms per record, then 3 ms under a
sanitizer, then 30 ms everywhere. Each was calibrated on a sample of one box and
each was afterwards found redding an unmutated tree. The 30 ms also cost
resolution: it widened the smallest swallow (1c) can see from about 0.5 ms to
60 ms.

#### The red-before, on an UNMUTATED tree at `73ada0df8`

120 runs of the containment case alone, one Release CPU-only binary, one build
directory, this 20-core box at loadavg 40-155, full output kept, 1440
leaf-record observations. **One run red.**

```
b3_003  loadavg 155.65 133.50 102.10
test_ltx2_video.cpp:4074: CHECK( span_slack <= span_bound )     0.0675591 <= 0.03
test_ltx2_video.cpp:4122: CHECK( covered >= c.min_coverage * leaf_seconds )
                                                            0.00730277 >= 0.0374371
[doctest] test cases: 1 | 0 passed | 1 failed
```

Both failures are **one event**. `artifacts.frames` render 2 record 2 runs
[55.431342293, 55.499995634] -- 68.65 ms -- and its `artifacts.frames.ppm`
sub-scope runs [55.431346873, 55.432441132]. The 67.55 ms between
`ppm_phase.Close()` and `write_phase`'s destructor holds two statements
(`rendered_frames += chunk.frames.frames` and the trace increment) and one
descheduled thread. It reds (1c) at 2.25x the 30 ms constant, and it reds (2)
because 67.55 ms of a 74.87 ms leaf is then uncovered.

That is the cluster in one observation: the SAME scheduler event decides (1c)
and (2), on a tree nobody touched, and a reader of either red learns nothing
about the tree.

#### The design: the denominator is measured, and it is measured FOR the render

A boundary is not a constant, so the bound is not one either. What one costs is
read out of `render_phase_log.cpp` rather than guessed: `PhaseLog::Open` takes
the process-wide mutex, stamps `o.start`, appends the open entry, calls
`SampleLocked` -- one `/proc/self/statm` read -- and, on the shipped default,
writes and FLUSHES one line to stderr. `Close` takes the same mutex, samples,
THEN stamps `r.end`, and flushes its own line. Both stamps sit inside that work,
which is exactly what puts the cost inside the leaf record.

`InstrumentCeiling` times those three operations, through the instrument's own
public entry points, from a thread that runs FOR the load and both renders at
one sample per millisecond, and the bound reads the WORST it saw.

**The worst, and not a mean, and that is the design rather than a detail.** The
numerator is itself a maximum over a small number of draws. A mean denominator
under a maximum numerator is the shape `ltx25-phase-residue.md` `## Design` 3
measured at 4 reds in 45 runs -- worst ratio 4.115 against a bound of 1 -- and
withdrew, because under contention the un-instrumented remainder of a boundary
dilates faster than the instrumented part, so the comparison has a heavy right
tail rather than a shifted median. Here both sides are tail quantities.

**Two denominator shapes were measured and rejected**, and both are worth
carrying forward because both are the obvious thing to reach for:

| ceiling shape | worst `slack / ceiling` over 1440 observations |
|---|---:|
| bursts of 4096 taken BESIDE each render (about 40 ms of a case that runs for tens of seconds) | **2.954** |
| the continuous sampler's 99.9th percentile | **199.4** |
| **the continuous sampler's maximum** | **0.645** |

The 99.9th percentile is the interesting reject. It is 20x to 200x tighter and
would be a far better instrument if it were safe. It is not, for the same reason
the mean is not: it is not a maximum. On run b3_003 it read 0.339 ms against a
67.55 ms boundary, so it would have kept the very red this change exists to
explain.

#### The bound, and the whole of the judgement in it

```cpp
const double kBoundariesPerRecord = 4.0;
const double kSpanSlackPerRecord = kBoundariesPerRecord * ceiling.ceiling_seconds();
const double span_bound = std::min(kSpanSlackPerRecord, 0.5 * record_seconds);
```

The head of a leaf record is `Open(leaf)`'s sample and flushed line followed by
`Open(sub)`'s acquisition of the same mutex; the tail is `Close(sub)`'s flushed
line followed by `Close(leaf)`'s acquisition, sample and stamp. TWO
boundary-sized operations. `2 x ceiling` is what the structure gives; the four
carries one factor of two over it, and that factor is the only number in the
line that was chosen rather than derived. Against the measured population it is
6.2x inside the worst observation, and zero of 1440 records violate it at any
`C` from 1 upward.

#### What it costs, and what it buys, measured on the same box

The bound is a property of the run, so its RESOLUTION is too. Measured over the
same 120 runs, `4 x ceiling` ran **1.18 ms to 419 ms, median 13.3 ms**, against
the flat 30 ms it replaces. On the quiet end that is 25x tighter; at loadavg 155
it says out loud that it can resolve nothing under a tenth of a second, which
run b3_003 proves is the truth about that box rather than a weakness of the
bound.

Staged as a mutation, the difference is visible directly. `D-unnamed-head`
inserts a phase nobody named, of a known size, between `Open("decode.audio")`
and `Open("decode.audio.mel")` -- the exact defect (1c) exists for:

| mutation | old bound (30 ms) | new bound |
|---|---|---|
| 20 ms un-named head | **GREEN on (1c)** (caught only by `decode.audio`'s 0.99 coverage floor) | **RED on (1c), 3 of 3**, at a run ceiling of 79-126 us and a bound of 0.32-0.50 ms |
| 2 ms un-named head | **GREEN 3 of 3** | RED 1 of 3 -- and the two greens report a ceiling of 2.26-2.56 ms, so the bound was 9-10 ms and the run said so |

#### The same term was written for (2), MEASURED, and WITHDRAWN

The obvious companion change is to give the coverage floor the same allowance as
a second arm:

```cpp
CHECK(covered >= c.min_coverage * leaf_seconds || uncovered <= boundary_allowance);
```

with `boundary_allowance = 2 * (leaf records + sub-scope records) * ceiling`. It
was written, built, and it makes b3_003's coverage red green. **It is withdrawn,
because it makes a real mutation green as well.**

Mutation `B-empty-ppm` moves `phase::Scope ppm_phase("artifacts.frames.ppm")`
from before the `WriteFileBytes` loop to after it, so the writer's anchor stops
covering the writes -- the defect that anchor exists for. Coverage of
`artifacts.frames` falls from about 98% to **1.1% and 5.0%**. Under the
disjunction the case reported `carried by the INSTRUMENT ARM alone` and passed,
3 of 3. Against the floor as it stands it reds, 2 of 2.

The reason is a property of the fixture and not of the arithmetic:
`artifacts.frames` is **0.2 ms to 7 ms** here, while one boundary on this host
is **0.3 ms to 1.2 ms even pinned to two otherwise idle cores** (`taskset -c
18,19`). Any allowance built out of the boundary is larger than the leaf. The
0.50 floor on that leaf is a RELATIVE test at a scale the absolute instrument
cannot reach, which is why it catches `B-empty-ppm` and why it reds under load:
those are the same property, and no threshold separates them.

So (2) is unchanged by this row, and #1470 keeps it. What would settle it is an
anchor whose seconds the writer can be read against -- the same class of repair
`### Owed out of W0` already asks for on the short records, and the same class
`denoise.update` ([#1668](https://github.com/mudler/vllm.cpp/issues/1668)) is
for `denoise`. It is a production scope, not a number.

#### Mutations

Restored byte-for-byte after each, `git status` clean between runs; every
mutation prints `compile_status=0` and asserts its anchor matched exactly once,
so a mutation that failed to build or failed to apply cannot read as a pass.

| Mutation | What it stages | Verdict |
|---|---|---|
| `C-no-sampler` | the `CeilingSamplerGuard` is deleted from the case, so every bound would be `4 x 0` | **RED**, `REQUIRE( ceiling.observations() > 0 )`, 45 assertions in |
| `D-unnamed-head-20ms` | a 20 ms phase nobody named in `decode.audio`'s head | **RED on (1c), 3 of 3** (green on (1c) at the 30 ms constant) |
| `D-unnamed-head-2ms` | the same at 2 ms | RED 1 of 3, and the greens print the ceiling that explains them (green 3 of 3 at the 30 ms constant) |
| `B-empty-ppm` | `artifacts.frames.ppm` opened after the write loop instead of around it | **RED on (2), 2 of 2** -- detection unchanged by this row, which is why the (2) arm was withdrawn |

#### What this does NOT do, stated before it is relied on

* **It does not close #1439.** `leaves >= 0.95 * wall` is untouched and is still
  a share of a contended clock.
* **It does not close #1470.** The coverage floors still decide by load on
  `artifacts.frames`, at a measured rate of 1 run in 120 on this box, and the
  section above is the measurement that says why a bound cannot fix it.
* **#1536 STAYS OPEN. Its thesis is contradicted by the population below, and
  "refuted rather than repaired" was written on the assumption that this shape
  landed. It did not.** That issue's claim is that
  the suite is "persistently red on an idle box, not load-flaky". At
  `73ada0df8` it is the opposite: 119 of 120 green, and the single red was at
  loadavg 155.65. Its 0.18-0.8% misses were against the 0.95 floor that
  `6b48edb2c` moved to 0.75; the measured coverage on this box is 99.6% for
  `denoise` and 98.4%-100% for the decode leaves, so nothing is near that floor.
* **The sampler perturbs what it measures.** It takes the phase mutex about a
  thousand times a second against a render that takes it about thirty times in
  total, and it writes to the same fd the live lane writes to. The perturbation
  is inside the quantity it reports, which is the conservative direction. It is
  also why the guard is RAII: a failing `REQUIRE` in this case would otherwise
  leave a thread writing to fd 2 while the later cases in the file `dup2` a
  capture file over it.


### The span-slack bound, FIFTH and SIXTH shapes: both MEASURED, both WITHDRAWN (#1572, #1576, #1536 stay open)

**Nothing in this section lands.** (1c) on `main` is unchanged and still carries
the fourth shape's 30 ms constant. What follows is why two further shapes were
built, measured and withdrawn, so the next person does not build them again.

| shape | what the denominator was | how it fails | measured |
|---|---|---|---|
| 4th (on `main`) | a flat 30 ms per leaf record | reds an unmutated tree under load, and cannot resolve `artifacts.frames` at all | [#1576](https://github.com/mudler/vllm.cpp/issues/1576), 171x on one binary |
| 5th | `4 x` the WORST boundary a 1 kHz sampler saw across the WHOLE case | one scheduling event on the sampler thread multiplies the bound by ~100 and silences a real 20 ms un-named phase | fresh review: defect reds 9/9 alone, 0/4 with one 200 ms sampler stall, case GREEN 2/4 |
| 6th | `4 x` the worst boundary inside the RECORD'S OWN window | too tight: the honest span slack's tail exceeds it, because the UN-instrumented remainder of a boundary is in the numerator and not in the denominator | **10 red in 45 consecutive runs** of the unmutated containment case, loadavg 21.8-61.5 -- of which **5 in 45 is the defensible figure**, because a second mechanism was repaired while the population was still running and its reds were measured on a binary that predates its own repair; see the note under the population below |

The fifth shape's derivation is right and a fresh review confirmed it
operation-by-operation against `render_phase_log.cpp`. **Its ESTIMATOR was
wrong, and the review broke it by execution rather than by argument.**

#### The defect: an extreme-order statistic compared across sample sizes

`ceiling_seconds()` was the maximum over every draw the sampler took across the
whole case -- **3,764 to 27,975 draws** per run -- while the numerator it bounds
is a maximum over **TWO**. Maxima do not compare across sample sizes. Over 24
unmutated runs the reviewer measured the sampler's MEDIAN operation at
**18.4-22.9 us, stable to 1.24x across loadavg 25 to 82**, and its MAXIMUM at
**47x to 785x that median**. A single scheduling event on a thread that does
none of the render's work therefore multiplied the bound by about a hundred.

The consequence is a silent green, staged and measured:

| arm | what it stages | (1c) red | case verdict |
|---|---|---|---|
| `C20` | a 20 ms un-named phase in `decode.audio`'s head | 9 of 9 | red 9 of 9 |
| `C20D2` | the SAME defect plus ONE 200 ms descheduling of the sampler thread | **0 of 4** | **GREEN 2 of 4**, `assertions: 603 \| 603 passed \| 0 failed`, bound 0.800 s |

A real un-named phase, no assertion fires, and the only thing that made it pass
is a scheduling event on a sleeping background thread. `V15D2` reproduces it on
the leaf where (1c) is the ONLY detector: `V15` reds (1c) 5 of 5 and `V15D2`
reds it 0 of 2.

#### The repair is not a smaller multiplier

Each draw is stamped with `PhaseLog::Elapsed()`, which is the very clock
`start_seconds` and `end_seconds` are on, and a leaf record's bound reads only
the draws that fall **inside that record**. The denominator is then drawn from
the same interval as the numerator.

It is a **strict tightening**: a maximum over a subset is never larger than the
maximum over the whole, so no record's bound grew. Measured green on this box at
loadavg ~80, over both renders of the fixture:

| leaf | record | bound, sixth shape | bound, fifth shape (whole-case) | flat constant it replaced |
|---|---|---|---|---|
| `artifacts.frames` | 4.33 ms | **71.9 us** | 6.1-20 ms | 30 ms -- **could never resolve this leaf at all** |
| `decode.audio` | 2.31 s | **0.601 ms** | 6.1 ms | 30 ms |
| `decode.video` | 0.224 s | **0.495 ms** | 6.1 ms | 30 ms |
| `denoise` | 7.92 s | **6.15 ms** | 6.1 ms | 30 ms |

`artifacts.frames` is the leaf [#1470](https://github.com/mudler/vllm.cpp/issues/1470)
is about, and it is the first time any shape of this bound has been able to
resolve it.

A preemption INSIDE a record still widens that record's bound, and that is
correct rather than a residual defect: if the box descheduled this thread by
200 ms inside this leaf then this leaf's own slack could honestly be 200 ms, and
(1c) is entitled to say it cannot resolve that. What it may no longer do is stay
silent about it -- see the next section.

The same change retires two smaller defects the review found. The whole-case
ceiling is MONOTONE, so `CheckRenderPhases`'s third call re-checked render 1's
records against a bound **1.00x to 11.59x larger** than the first call's -- the
same records at two strictnesses in one run. And the bound and the number
printed beside it were two separate reads of a moving quantity, so the report
could disagree with the arithmetic it described. A per-record window fixes both
by construction.

#### The skip path was a permanent report-only state, which `AGENTS.md` refuses -- and the assertion that answered it is NOT in the tree

The observation holds against `main` and is not about the withdrawn shape:
`span_unresolvable` (`test_ltx2_video.cpp:3972`) and `span_checked` (`:3983`)
are computed, printed once in a `MESSAGE` (`:3999-4001`) and **asserted
nowhere**, and the lane that matters discards the printing: CI runs
`ctest --output-on-failure`, so on a green run nobody ever sees the line. The reviewer measured four
`CheckCarryingPhase` calls reporting `1 of 1`, `0 of 2`, `0 of 1`, `0 of 1`
while the case exited `Status: SUCCESS!` with a 20 ms un-named phase present.
`AGENTS.md` `## Gates` refuses that in as many words: *"Report exactly one result
for each applicable rule ... A permanent report-only state is not a result."*

The skip condition that goes with it is `main`'s, not the withdrawn shape's:
a record is checked only when `record_seconds >= 2 * kSpanSlackPerRecord`, the
flat 60 ms, which is what `span_bound < kSpanSlackPerRecord` tests at
`test_ltx2_video.cpp:3971`. The `8 x` form below -- a record against its OWN
window's worst boundary -- belongs to the withdrawn per-record window and never
landed.

**The assertion the withdrawn shape added was `longest_checked`** -- (1c) must
have resolved the LONGEST record of each carrying leaf -- **and it is NOT in the
tree**, because it went out with the shape it was built on. It named a record
rather than a count or a share, which is what kept it free of a new constant:
the longest record is the one that can hide the most AND the one most likely to
satisfy `record >= 8 x its own window's worst boundary`, so the two orderings
agreed and demanding it was both the strongest and the cheapest thing to demand.
A count floor would be a constant; a seconds share would be a constant; "the
biggest one" is neither. **So the report-only state above is still open on
`main`**, and the argument for this shape of assertion is what survives, not the
assertion.

In that shape a record too short to hold a single draw of a 1 kHz sampler was
counted and reported separately from a record the bound could not resolve,
because those are different facts about the instrument. That separation is not
in the tree either; `main` counts `span_unresolvable` alone.

#### Two more findings, repaired in the withdrawn shape, NONE of which is in the tree

* **The sampler's flushed write went to fd 2 at 1 kHz and corrupted the live
  lane.** One capture held 158 `[render]` occurrences of which only **116 began
  a line**, so `grep '^\[render\]'` silently lost 27% of them -- and #1413 is the
  row that exists for that stream. The same review deleted the write entirely and
  the estimator did not move (green 3 of 3, ceiling 3.24-9.19 ms against an
  unmutated range of 0.94-17.5 ms), so the SINK is not what the term measures.
  In the withdrawn shape the write was kept and pointed at `/dev/null`. **That
  repair went out with the shape and is NOT in the tree**, so the `[render]`
  stream-corruption [#1413](https://github.com/mudler/vllm.cpp/issues/1413) names
  is not fixed by anything here; what stands is the measurement that the sink is
  not the term.
* **`p999_seconds()` was dead**, called from nowhere, and would not have done what
  its comment claimed: `Quantile(999, 1000)` clamps to `n - 1` for every
  `n <= 1000`, so at the sample counts a short run produces it returned the
  maximum. In the withdrawn shape the accessor was deleted; **that deletion is
  NOT in the tree either**, because `p999_seconds()` only ever existed on the
  withdrawn estimator. What is kept is the argument for rejecting a quantile.

#### And the sixth shape was measured too, and it is WITHDRAWN

The repair above was built, its whole mutation set was re-run against it, and it
detects better than anything before it: a **2 ms** un-named phase in
`decode.audio`'s head reds (1c) **3 of 3**, against 1 of 4 on the fifth shape and
0 of 4 at the 30 ms constant; a 15 ms one in `decode.video`'s head reds 3 of 3;
and the 20 ms defect with the 200 ms sampler stall that silenced the fifth shape
reds **3 of 3**.

**And then it was validated the way this cluster's own rule demands, and it
failed.** 45 consecutive runs of the containment case, one pinned binary
(sha256 `242c7925...`), an unmutated tree, this 20-core box at loadavg
**21.78 to 61.52**, every run's counts recorded:

```
45 runs   35 green   10 RED   (22%)
```

**THE HEADLINE 22% IS NOT THE RED RATE OF THE SHAPE'S FINAL FORM, and the honest
figure is the 5.** Two mechanisms produced those ten, and the second one was
identified and repaired while the population was still running -- so its nine
reds were measured on a binary that predates its own repair, and that repair was
never re-validated. **11%, five reds in 45, is the defensible number**, against
a measured 1 red in 120 (0.83%) for what is on `main`. It is 13x, it is decided
by the same mechanism at any multiplier, and it is what the withdrawal rests on.
The nine are reported rather than dropped because a population that reports only
its surviving mechanism is an argument and not a measurement.

Two mechanisms, both honest:

| assertion | reds | what happened |
|---|---:|---|
| `span_slack <= span_bound` | 5 | the honest head-and-tail exceeded `4 x` the record's own local worst boundary -- `decode.audio` 13.115 ms against 8.889 ms, `decode.video` 0.488 ms against 0.374 ms, `artifacts.frames` 70.6 us against 60.6 us |
| `longest_checked` | 9 | a leaf's longest record was not resolvable, most often `artifacts.frames` at about 1 ms with no draw of a 1 kHz sampler inside it |

**Those two counts do not sum to the ten red RUNS, and this record cannot close
the gap.** 5 + 9 = 14 against 10, so the two columns overlap; the run-by-run
counts behind `45 runs / 35 green / 10 RED` were recorded during the population
but are not reproduced in this file, and nothing here says whether a cell counts
red RUNS or red ASSERTION INSTANCES. Read as runs the overlap is forced to be at
least four, because 14 events cannot be spread over 10 runs without four
doubling up; read as instances one run can contribute two reds on two leaves and
the overlap is not bounded at all. **The split is therefore an unrecorded
quantity and no number here should be read as one.** What is load-bearing, and
what the withdrawal rests on, is the 10 red runs in 45 and the 5-in-45
defensible figure -- neither of which needs the split.

The first mechanism is the one that matters and it is the SAME one
[`ltx25-phase-residue.md`](ltx25-phase-residue.md) `## Design` 3 records for the
withdrawn residue bound: **the part of a boundary the instrument cannot measure
-- the `lock_guard` release, the `Close` return, the `Scope` destructor and
constructor, the call into `Open` up to its clock read -- dilates faster under
contention than the part it can.** It sits in the numerator and in no
denominator any sampler can build, so tightening the denominator onto the
record's own window moves the bound INTO the honest distribution's tail. Reads
against it are then decided by the scheduler, which is what all four previous
shapes were.

The second mechanism was repaired mid-flight -- a record shorter than the
sampler's cadence now takes the worst of the draws bracketing it -- and the
first is not repairable by choosing a different denominator.

**So the estimator is withdrawn and (1c) is left exactly as `main` carries it.**
Raising the multiplier would be repairing a red gate by widening it, which
`AGENTS.md` `## Changing the rules or a checker` refuses in as many words.
[#1668](https://github.com/mudler/vllm.cpp/issues/1668) does not say that and is
not cited for it -- what it says, quoting this file's own comment, is that
"NAMING THE UN-NAMED TIME WOULD SETTLE IT PROPERLY", i.e. that the repair is an
ANCHOR and not a threshold, which points the same way by a different route. Four
shapes have now been tried and the
common factor is that all four compare a wall-clock quantity the scheduler owns
against a threshold the tree owns.

#### The next traceable hypothesis, which this row does NOT take

A defect is REPRODUCIBLE within a run and a scheduling event is not. The fixture
renders the same three carrying leaves TWICE, and `CheckRenderPhases` is called
for both passes. A bound that reds only when the SAME leaf exceeds it in BOTH
renders would keep every detection measured above -- the injected sleeps run on
every pass -- while a one-off preemption, which is what all ten reds above are,
hits one pass and not the other. In the five span-slack reds the failing leaf
exceeded the bound in exactly one render.

**AND ITS COST IS NAMED BEFORE ANYBODY BUILDS IT.** Requiring both renders
silences any real defect present in only ONE of them -- a first-render lazy
allocation, a cold-cache path, anything the second pass warms. On this fixture
the three carrying leaves run the same code on both passes, which is why the
staged sleeps red 3 of 3, but that is a property of the fixture and not of the
assertion. Whoever builds it owes a mutation that is present in render 1 alone,
and has to say what happens to it.

That is a different assertion with a different failure mode and it owes its own
red-first evidence, its own mutation set, and its own 45-run validation. It is
recorded here rather than attempted, because attempting a fourth shape in the
same session that refuted the second and third is how this cluster got four
constants in the first place.

#### What is still owed after this shape

* **On this box the fifth shape did not buy back the resolution it claimed.**
  The reviewer measured the bound looser than the 30 ms constant in **4 of 24
  runs**, and records resolvable at **212/408 (52.0%)** on the fifth shape
  against **204/408 (50.0%)** at the constant -- 1.7x at the median, not the 25x
  the section above claims. The sixth shape's per-record windows are what change
  that; the fifth shape's "25x tighter on the quiet end" line is left standing
  above as what was measured on ITS population and should be read against this
  one.
* **The sanitizer lane is untested for this estimator.** The mechanism the third
  shape's per-build constant existed for is inherited automatically, because the
  sampler compiles into the same binary -- but the sample count also grows with
  the slowdown. Nobody has run it.
* **The ROCm arm is a hypothesis, not a measurement.** `FixtureParams` sets
  `mp.device = 0`, `ltx2_video.cpp` installs a `DeviceByteProbe`, and
  `src/vt/rocm/rocm_backend.hip` implements `DeviceMemoryInfo` where
  `CudaBackend` does not. On a ROCm runner this sampler would drive
  `hipMemGetInfo` at 1 kHz under the phase mutex. No GPU lease was taken.
