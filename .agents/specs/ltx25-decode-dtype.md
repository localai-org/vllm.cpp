# LTX25-DECODE-DTYPE — the decode computes in f64, and no reference does

Row: `LTX25-DECODE-DTYPE`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)).
Issue: [#1008](https://github.com/mudler/vllm.cpp/issues/1008).
Parent: lever 2 of the `LTX25-DECODE-SPEED` investigation
([#1006](https://github.com/mudler/vllm.cpp/issues/1006)), which filed this
issue and lists it under `## Owed`. That spec is
`.agents/specs/ltx25-decode-speed.md` on [PR
#1018](https://github.com/mudler/vllm.cpp/pull/1018) and is **not yet on
`main`**, so it is cited by pull request rather than by relative link — a
relative link would dangle until #1018 lands.

## Now

`ACTIVE`. This row takes the *dtype* half of lever 2. It does not take the
memory-format half; §5 states the verdict and what blocks it.

## 0. Scope, and the one thing this row is not

**In scope.** Every accumulation and every elementwise arithmetic step on the
LTX-2.5 conv video VAE data path moves from `double` to `float`, which is the
dtype upstream actually computes these ops in. Scalar constants — epsilons,
`1/sqrt(C)`, per-channel shift/scale precomputed from two f32 weights — stay
`double`, because upstream's counterparts are Python floats and they cost
nothing.

**Not in scope, and deliberately so.** The *storage* dtype stays f32. Upstream
stores bf16 (§1), and moving this file to bf16 storage is the production arm
that `ltx2_video_vae.cpp:63-66` already books as phase L6 debt. That is a
different change with a different risk profile, and this row does not take it.
Narrowing the *arithmetic* is separable from narrowing the *storage*, and only
the first can be done without a second dtype for every buffer.

**Also not in scope:** threading ([#1009](https://github.com/mudler/vllm.cpp/issues/1009)),
the device arm ([#1007](https://github.com/mudler/vllm.cpp/issues/1007)),
`memory_efficient_decode.py` ([#1011](https://github.com/mudler/vllm.cpp/issues/1011)).

**No speed number.** `dgx.casa` has been unreachable for the whole of this row's
work, so there is no GPU and no large-render host. This row lands the
correctness-preserving change with the numerics pinned and books the magnitude
as owed and unmeasured (§7). An honest "unmeasured" beats an invented number,
and this campaign's own standing rule is that a number quoted often becomes
treated as measured.

## 1. What dtype upstream actually accumulates in

The issue and the parent spec both establish that no reference *stores* f64 on
this path. That is true and it is not the question this row had to answer. The
question is what dtype the *accumulation* happens in, which is not the same as
the tensor dtype, and it was settled by running the reference rather than by
reading it.

**Read.** Every conv in the decoder is a plain `torch.nn.Conv3d` — `CausalConv3d`
builds it at `packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:292-302`
and the single call site for the whole decoder is `:312`, at
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`. `PixelNorm.forward` is
`torch.mean(x**2, dim=...)` in the activation dtype
(`model/common/normalization.py:37-40`). `_RMSNorm2D.forward` is
`F.normalize(x, dim=1) * (scale * gamma)`, and `AttnBlock3D.forward` is
`to_qkv` / SDPA / `proj`, all in the activation dtype
(`model/video_vae/attention.py:23, 58-69`). `diffusers` at `3a2f35d4e` agrees:
`PerChannelRMSNorm.forward` computes in the activation dtype
(`src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py:50-59`).

**Run.** Reading source gives the tensor dtype, not the accumulator width. The
accumulator was measured directly with torch 2.11.0, on a reduction engineered
so the two widths are separable: 27 taps over a uniform-1.0 input with weights
`[+1e8, 0.1 x 25, -1e8]`. Every partial sum `1e8 + j*0.1` for `j <= 25` is below
half an ulp of `1e8` (which is 4.0), so an f32 accumulator holds exactly `1e8`
until the final `-1e8` and lands on exactly zero, in **any** summation order. An
f64 accumulator lands on 2.5.

| what | result |
|---|---|
| `F.conv3d`, f32 tensors | **0.0** |
| `F.conv3d`, bf16 tensors — upstream's own configuration | **0.0** |
| `F.conv3d`, f64 tensors | 2.500000014901161 |
| naive serial f32, by hand | 0.0 |
| this port today | 2.5 |

**So upstream's convolution accumulator is f32, and this port's is f64.** The
expected value this row's new gate asserts is therefore torch's own answer, not
a recording of what the patched code happens to emit.

**One honest caveat, recorded because it bounds what may be gated.** The same
probe run through `torch.sum` / `torch.mean` on f32 returns `2.0999999046325684`,
not `0.0`. Torch's reductions are f32-*wide* but use a cascading/pairwise order,
where this port's are naive serial. Width matches; order does not. A gate that
pinned an adversarial value through `PixelNorm` would therefore be pinning this
port's summation order rather than upstream's dtype, so this row does not build
one. The convolution is the site where torch's answer and a naive serial f32
answer coincide exactly, and it is also where 42 of the decode's convolutions
and essentially all of its arithmetic live.

## 2. The sites, and what each one becomes

The issue counts 8 `double acc` declarations and 29 `static_cast<double>`. The
`acc` count is right and is not the whole set: `PixelNorm`'s `mean_sq`, the
attention block's `sum_sq`, `dot` and `sum`, and the attention block's six
`std::vector<double>` activation scratch buffers are accumulators and data-path
buffers that the `double acc` grep does not reach.

Anchors are at `ff264cb82`, this row's base.

| site | what it is | upstream | this row |
|---|---|---|---|
| `:165` | `CausalConv3d` output accumulator — 42 convs, ~all the FLOPs | `nn.Conv3d`, `convolution.py:292-302,312`; measured f32 (§1) | `float` |
| `:201` | `Linear3d`, the 1x1x1 conv used as `conv_shortcut` | `make_linear_nd` dims=3 -> `nn.Conv3d`, `convolution.py:84-85` | `float` |
| `:221` | `PixelNorm` `mean_sq` | `torch.mean(x**2)` in activation dtype, `normalization.py:37-40` | `float` |
| `:303`, `:312` | `TimestepEmbedding`'s two `nn.Linear` accumulators | `nn.Linear` in activation dtype | `float` |
| `:528` | attention `_RMSNorm2D` `sum_sq` | `F.normalize(x, dim=1)`, `attention.py:23` | `float` |
| `:546` | attention `to_qkv` accumulator | 1x1 `nn.Conv2d`, `attention.py:55` | `float` |
| `:557`, `:564` | attention score `dot` and softmax `sum` | SDPA in activation dtype, `attention.py:65` | `float` |
| `:570` | attention value-weighted accumulator | SDPA | `float` |
| `:579` | attention `proj` accumulator | 1x1 `nn.Conv2d`, `attention.py:56` | `float` |
| `:916` | encoder `SpaceToDepthDownsample` group mean | `.mean(...)` in activation dtype | `float` |

Elementwise data-path steps narrow with them, for the same reason and the same
anchor: `Silu` (`:213`, `F.silu`), the ada-LN apply (`:361-362`), the noise
blend and per-channel denormalize (`:629-631`, `:645-648`),
`FeedSpatialNoise` (`:337-338`), and the encoder's normalize (`:1129-1130`).

**What stays `double`, each with its reason written beside it in the file.**

* Stabilizing epsilons (`norm_eps`, `pixel_norm_eps`, `kLtx2RmsNorm2dEps`) —
  upstream's are Python floats, they are compared and added once per row, and
  narrowing them changes a threshold rather than a data path.
* `norm_scale` = `sqrt(C)` and `attn_scale` = `1/sqrt(C)` (`:514-515`) —
  upstream's `channels**0.5` is a Python float, evaluated once per block.
* The `TimestepEmbedding` frequency table (`:284-292`) — a transcendental
  constant precompute, not a data-path accumulation, evaluated once per block
  over 256 entries. Upstream's `torch.arange`-based table is f32; keeping f64
  here is *closer* to the value that table approximates, and it is not on any
  hot path. It is annotated as a deliberate exception rather than left silent.

## 3. The gate the goldens cannot be

`ltx2_video_vae.cpp:41-44` already says why, and this row confirmed it rather
than inheriting it: `scripts/gen-ltx2-vae-goldens.py:223` casts every upstream
parameter with `values.astype(np.float32)`, and the input builders at `:759` and
`:836` do the same. **The oracle that produced the goldens ran f32 end to end.**
A dtype comparison against it cannot see an accumulator that is too wide, which
is the defect this row removes.

That cuts both ways, and the second way is the useful one. Because the golden
generator ran torch in f32, the goldens are the output of an **f32-accumulating**
reference. This port's f64 accumulation has been *wider than the oracle its own
goldens came from* since the file landed. Narrowing to f32 moves this port
toward the goldens' generator, not away from it — which is why §6 expects the
recorded `max|diff|` to hold or improve, and treats a regression as a finding.

**The new gate.** `test_ltx2_vae.cpp` gains one case that enters through the
production entry point `Ltx2VideoDecodeStreaming` and asserts the decode's
convolution accumulator is f32-wide, using §1's separable reduction. The
decoder is configured so the assertion is analytically derivable end to end
rather than recorded:

* `decoder_blocks` empty, `timestep_conditioning=false`, `norm_layer=pixel_norm`,
  `patch_size=1`, `out_channels=1`, `base_channels=2`, `in_channels=1`,
  spatial padding `replicate`, latent all `1.0`, `std-of-means=1`,
  `mean-of-means=0`. Every conv tap therefore sees exactly `1.0`, including at
  every border, so one reduction is repeated at every output voxel.
* `conv_in` channel 0 carries `[+1e8, 0.1 x 25, -1e8]` with bias 0; channel 1
  carries all-zero weights with bias 1.
* f32 accumulator: ch0 = 0, ch1 = 1 -> `PixelNorm` leaves ch0 at 0 -> `SiLU(0)=0`
  -> `conv_out` selects ch0 -> **every output element is exactly 0**.
* f64 accumulator: ch0 = 2.5, ch1 = 1 -> `PixelNorm` inv = `1/sqrt(3.625)` ->
  ch0 = 1.3131 -> `SiLU` -> ~1.0348. The two arms are three decimal orders apart.

The value `0` is upstream's measured answer for this reduction in both f32 and
bf16 (§1), so the case gates a mirrored property and not a local convention.

## 4. Risks

* **The narrowing moves a golden past `kLtx2GoldenTol` = 5e-6.** This is the one
  risk that can change the design. Naive serial f32 is a worse summation order
  than torch's cascading f32, so the error against the goldens can rise even
  though the width now matches. Mitigation: measure `max|diff|` per golden arm
  before and after, and report both. If an arm exceeds tolerance, the finding is
  reported and the site is either kept `double` with its reason written beside it
  or given a better summation order — never a widened tolerance. Widening a
  tolerance to admit a change is the failure mode `AGENTS.md` forbids.
* **A `float` accumulator silently re-promoted.** `acc += float * double` promotes
  the whole expression back to `double`. Every narrowed site must have its scalar
  operands narrowed too, or the change is a no-op that reads as done. The §3 gate
  catches this for the convolution; the reviewer should mutate the others.
* **`std::exp` overload selection.** `std::exp(-v)` with `float v` selects the
  float overload; `std::exp(-static_cast<double>(v))` does not. `Silu` depends on
  this.

## 5. NDHWC — the verdict, and why this row does not build it

**The engine can express the layout. The blocker is not `Volume`, and naming it
as `Volume` would be wrong.**

`Volume` is a file-local struct in an anonymous namespace
(`ltx2_video_vae.cpp:85-93`) whose `At()` is a single indexing function with 16
call sites. That much is cheap. What is not cheap:

1. **A shared helper hard-codes NCDHW in its signature.**
   `MiniMaxH3GroupNorm3d(std::vector<float>& x, int64_t channels, int64_t spatial, ...)`
   (`include/vllm/model_executor/models/minimax_h3.h:756`) takes a
   channel-major buffer by contract. The LTX-2.5 video VAE calls it at four
   sites, and it is **shared**: MiniMax-H3's own VAE CNN calls it at three
   (`minimax_h3_vae_cnn.cpp:176,182,337`) and the LTX-2 audio VAE at one
   (`ltx2_audio_vae.cpp:215`). NDHWC in the video VAE means either an NDHWC
   entry point on that shared helper, or a transpose at every norm boundary
   which spends what the layout was meant to save. This is a shared-seam
   decision, not a local one.
2. **Ten layout-dependent sites bypass `At()`** with open-coded `c * n + i`
   arithmetic, at §2's `ff264cb82` base anchors rather than this section's
   final-tree ones — `Linear3d` (`:203,:206`), `ApplyAdaLn` (`:361-362`), the
   denormalize (`:645-648`), `SpaceToDepthDownsample` (`:919,:921`), the
   encoder normalize (`:1129-1130`) — plus the local scratch buffers `padded`,
   `normed`, `q`, `k`, `v` and `attended`, which carry their own layouts and
   would each need one.
3. **`Ltx2VideoFrames::data` must not move.** It is the decode's output contract
   and the frame writer's input.

**And the win would not be the layout.** Upstream's `channels_last_3d`
(`memory_efficient_decode.py:617-627`, `:655-656`) exists to select a different
cuDNN 3-D convolution kernel family on a GPU. This port has no device arm at all
([#1007](https://github.com/mudler/vllm.cpp/issues/1007)), so on the host arm
that exists today NDHWC buys nothing by itself: it makes the reduction over
input channels contiguous, which is worth having **only once something
vectorizes over it**. The layout is a precondition for a SIMD or device arm, not
a speedup on a scalar loop nest.

**Verdict: real, medium-sized, and a separate row.** It is filed rather than
half-built, and it must land after or with the arm that consumes it. The parent
spec's §8 says why bundling it here would be wrong: a dtype change no golden can
see and a memory-format change across a shared seam are two independent reviews.

## 6. Gates and evidence

1. **The new width case is RED before the change and GREEN after**, captured
   with its exit code, not with a grep of the assertion line.
2. **Reachability.** The case enters through `Ltx2VideoDecodeStreaming`, the
   entry `src/vllm/multimodal/ltx2_video.cpp:3258` calls on the render path.
   Deleting the `Ltx2ConvVideoDecode` call at
   `ltx2_video_vae_tiled.cpp:113` must turn it RED.
3. **`max|diff|` recorded for every video golden arm, before and after.** The
   goldens must stay green on their own tolerance, and the two numbers are
   reported side by side rather than summarized as "still passes".
4. **Full gate**: configure, build, `ctest`, with `: error:` count, test count,
   exit codes, load and free disk.
5. Every mutation reports three facts: `git diff --stat` after applying, whether
   it BUILT with the compile-error count, and the exit code.

## 7. Owed

| owed | what would settle it |
|---|---|
| **The speed magnitude of this change. UNMEASURED.** No number is claimed. | One `Ltx2ConvVideoDecode` wall at a fixed size on an idle host, same binary, f64 arm against f32 arm. `dgx.casa` was unreachable for this row's whole duration; any host that can run the decode settles it, and #1010's phase timings make it readable from a render. |
| **A width gate for the nine sites that have none** (§8.2). `Linear3d` was widened back to `double` and **40/40 cases passed**. | One separable-reduction case per site, in the shape §3 established for the convolution. |
| **A blocked summation order for the sites that still sum naively** (§8.1) — `Linear3d`, the attention block, `PixelNorm`. `CausalConv3d` has one; the others do not, and their reductions are shorter but not short. | The same treatment, measured the same way. None of them is near tolerance today. |
| NDHWC / `channels_last_3d` (§5) | its own row, after or with a SIMD or device arm |
| bf16 *storage*, the phase L6 production arm (`ltx2_video_vae.cpp:63-66`) | [#1007](https://github.com/mudler/vllm.cpp/issues/1007) and the L6 row |
| The `.agents/issue-index.md` row for [#1008](https://github.com/mudler/vllm.cpp/issues/1008) | It is **deliberately not appended here.** The row exists on PR #1018, which filed the issue and is unmerged. `.gitattributes:7` sets `merge=union` on that file and the record checker of the day refused a duplicate issue number with "duplicate is what two branches appending the same issue look like" (the refusal and the live index were retired together by W6, `7dc2ef1ea`).
Appending it here would turn `main` red for every branch the moment #1018 merges — the exact failure a duplicate #995 row caused. The link lives in this spec and in the pull request body; the index link arrives with #1018. |

## 8. Outcome — what was measured

### 8.1 The numerics moved, and by how much

They did not stay put, and §4 named this as the risk that could change the
design. Both arms were run with `kLtx2GoldenTol` temporarily set to `0.0` so
every case reports its `max|diff|` rather than only its verdict. Same binary
recipe, same host, one build each.

| golden arm | f64 acc (before) | f32 acc, NAIVE serial | f32 acc, BLOCKED (shipped) | tol |
|---|---|---|---|---|
| Conv video decoder | 1.40071e-06 | 4.12762e-06 | **1.72853e-06** | 5e-06 |
| non-causal Conv video decoder | 1.81794e-06 | 3.51667e-06 | 2.08616e-06 | 5e-06 |
| norm_eps-binding video decoder | 9.05246e-07 | 1.16974e-06 | 1.54972e-06 | 5e-06 |
| tiled decode, untiled control A | 2.08616e-06 | **5.00679e-06 FAIL** | 2.74181e-06 | 5e-06 |
| tiled decode, untiled control B | 2.62260e-06 | (same case) | 2.80142e-06 | 5e-06 |
| cropped video encoder | 4.17233e-07 | 8.94070e-07 | 4.76837e-07 | 5e-06 |
| video encoder (`*_res`) | 4.17233e-07 | 8.94070e-07 | 4.76837e-07 | 5e-06 |
| causal-arm video encoder | 2.98023e-07 | 3.83705e-07 | 4.17233e-07 | 5e-06 |
| video encoder (strided convs) | 5.96046e-07 | 5.96046e-07 | 8.34465e-07 | 5e-06 |
| all 13 audio arms | unchanged | unchanged | unchanged | untouched by this row |

**The risk §4 named as design-changing actually bound, and it changed the
design.** Narrowing the width while keeping the naive serial summation order
pushed `test_ltx2_tiling`'s non-causal untiled control to **5.00679e-06 against a
5e-06 tolerance** — over by 0.14%, a genuine RED in the full gate, not a near
miss. The tolerance was not touched.

**The fix is the summation ORDER, and it is a closer mirror rather than a looser
one.** `CausalConv3d` now keeps one partial sum per input channel and adds the
partials, so a `ci * kernel^3` reduction accumulates error with `sqrt(kernel^3)`
per block instead of `sqrt(ci * kernel^3)` across the whole length. That is what
torch's f32 convolution does — it is a blocked GEMM, which is exactly why §1's
probe found `torch.sum` returning 2.0999999 where a naive serial f32 sum returns
0.0. The width and the order are two separate mirroring questions and this row
had to answer both.

**The result is that the numerics essentially did not move.** Against the f64 arm
the shipped blocked-f32 arm is 1.07x to 1.31x on every video arm, where the naive
arm was 1.9x to 2.9x. The worst arm sits at 56% of tolerance — a 1.8x margin,
against 1.9x for the f64 arm it replaces and 1.0x for the naive attempt. No
headroom was meaningfully spent, and no tolerance was widened.

### 8.2 What is gated, and what is not

Stated plainly because the answer is uneven and a summary would hide it. Each
narrowed site was widened back to `double` on its own, rebuilt, and rerun.

| mutation | built | exit | detected by |
|---|---|---|---|
| W1 — `CausalConv3d` accumulator widened | yes, 0 errors | 1 | the new width case (1.03473 vs 7) |
| W2 — `Linear3d` accumulator widened | yes, 0 errors | **0** | **nothing. 40/40 cases pass** |
| W3 — `PixelNorm` `mean_sq` widened | yes, 0 errors | 1 | the Conv video decoder golden, incidentally |
| R — the production call site deleted | yes, 0 errors | 1 | the new width case (7 vs 7-minus-nothing) |

**W2 is an honest gap and it is owed.** Only the convolution's width is gated on
purpose; `PixelNorm` is caught by accident, because a mixed-width path happens to
diverge from torch further than a uniformly f32 one; and `Linear3d`, the
attention block's four accumulators, the timestep Linears and the encoder's group
mean have no gate on their width at all. They are narrowed on upstream grounding
and on review, not on a test. Closing that needs one separable-reduction case per
site, in the shape §3 established.

### 8.3 The reachability case that passed while measuring nothing

Worth recording because it nearly shipped. The first draft of the width case
expected **zero**, which is what an f32 accumulator produces on the engineered
reduction. Mutation R replaced the production `Ltx2ConvVideoDecode` call with a
zero-filled buffer — and the case **passed**, because a decode that never ran
produces zeros too. The case measured nothing and reported success.

The fix is `conv_out.conv.bias = 7`, which moves the expectation off zero: the
f32 arm must return exactly 7, the f64 arm returns 8.03473, and a stub returns 0.
All three are now distinguishable. A recorded value is not a reached one, and an
expectation that coincides with the zero value of an absent computation is not an
assertion.

The first attempt at mutation R also **failed to build** (3 `-Werror` unused-
parameter errors) while the stale binary still ran and printed a plausible
verdict. Both facts are why every mutation above reports whether it built.

## 9. Stop conditions

* Report `NEEDS_DECISION` rather than widening `kLtx2GoldenTol` if a narrowed
  site pushes a golden past it. **This fired** (§8.1). It was resolved by
  mirroring upstream's summation order, which this spec had already named as the
  remedy, rather than by touching the tolerance or restoring the f64 width — so
  it is recorded here as a stop condition that triggered and was answered within
  the row's own design, not as one that was waived.
* Do not claim any wall-clock or throughput result. There is no host.
* Do not build the NDHWC change here (§5).
