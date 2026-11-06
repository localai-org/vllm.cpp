# The DFlash2 on-card speculative-decode runtime — a design study

- Row: `TT-DRAFTER-STUDY` (study row; the deliverable is this note, not code)
- Sources: the `dflash2` bundle `mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4`
  @ `7855cf0d2982064350991bdb8e1ec1db82f914b3` (Apache-2.0), fetched to a scratch dir
  (`/tmp/df2-bundle/`, not committed); the drafter checkpoint
  `incoai/Qwen3.8-27B-DFlash2` @ `dedf8df68adfb1afeaf7b7480c0a0243108177b4` (config/README
  metadata only — no weights were downloaded).
- Provenance chain (from the bundle's `PROVENANCE.json`): model source
  `thin/src-cp81` @ `e3ab64576fb6ccca73e1bf12322b9702fada99e1`; plugin source
  `plugin-step-rebuild` @ `d564ae09a6fe1d63428251ec4c665e7dff8652ae`;
  tt_metal commit `547841b0109bf57f020355287841cd4f38eec98b`; runtime lock python
  3.12.14 / uv 0.12.11 / ttnn 0.79.0 / vllm 0.26.0+empty, with `constraints.txt` pinned
  by sha256 (`b04c14b9…`).
- Every behavioral claim below cites a file:line in the fetched sources. Line numbers
  refer to the scratch copies at the pinned revision.
- No performance claims of our own: the bundle's numbers are quoted as claims with
  their source lines, never as measurements we reproduced.

## 1. The DFlash2 mechanism (Q1)

DFlash2 is a block-diffusion drafter paired with a standard speculative verify. The
mechanism has four moving parts, all visible in `extra_code/qwen38/`:

1. **Draft (one block forward, not K steps).** `DFlash2DrafterTP.draft`
   (`dflash2_tp.py:1053`) stages a block of `R = U*block` rows — user-major, row
   `u*block + j` — and runs the drafter's 5 layers once. The block is
   `[anchor, MASK x (block-1)]` (`_stage_draft`, `dflash2_tp.py:1027`): one real token
   (the anchor, the target's last confirmed token) plus mask ids. The drafter is a
   5-layer transformer with its own paged context KV over the *target's* residual taps
   (`fc → hidden_norm → layers`, `dflash2.py:532-543`).
2. **Verify (the target's own traced forward).** The verify chunk
   `[pending, d_0..d_K-1]` at positions `p+1..p+K+1` is the *target model's* forward,
   captured once and replayed (`model.verify_traced`, `model.py:2549`). The verify
   argmaxes on device into `cfg.ids_out` — greedy acceptance needs no logit readback.
3. **Accept (host, cheap).** Greedy compares ids (`_accept_greedy`,
   `spec_decode.py:431`); sampled runs exact rejection sampling over the verify logits
   (`SpecSampler.accept`, `spec_sampling.py:280`) with the classic
   `u_j < p_j(d_j)` test, the recovery draw from `norm(max(0, p − q))`
   (`_recover_q`, `spec_sampling.py:249`), and one uniform per draft drawn up front
   (`spec_sampling.py:329-331`) so the RNG stream is independent of where the rejection
   lands. The README's exactness claims follow from this: greedy is token-for-token
   because the accept rule *is* the target's argmax; sampled is distribution-exact
   because rejection sampling with the target's own `p_j` is the standard lossless
   construction (`README.md:38-42`).
4. **Extend.** The committed positions' tap rows go into the drafter's context KV
   (`extend_context`, `dflash2.py:582`), so the next draft attends the whole confirmed
   history plus the just-accepted block.

The substrate (`SpeculativeDecoder`, `spec_decode.py:39`) is drafter-agnostic: the
native MTP head and DFlash2 plug into the same `generate` loop (`spec_decode.py:679`,
one `while any(not d for d in done)` iteration at `spec_decode.py:906`). DFlash2's
difference is in the drafter: a block-diffusion forward that proposes all K candidates
in one pass from residual taps, instead of K chained MTP decode steps.

## 2. The on-card loop (Q2): "host touches each decode step once"

The serving form is `DFlash2DualBucketDecoder.step` (`dflash2_serving.py:898`). One
step is:

- host stages per-row inputs into persistent, fixed-address buffers
  (`_draft_inputs`, `dflash2_serving.py:927`; the packed-state columns are documented
  at `dflash_devloop.py:163` and the module docstring of `dflash_devloop.py`);
- replay the drafter trace, the verify trace, and the extend trace — all captured
  (`dflash2_serving.py:942-963`, `dflash_devloop_run.py:660-691`);
- host reads back ids and runs the accept math (`dflash2_serving.py:978-1005`).

So the host's per-step work is: stage small tensors, replay, accept. The heavy lifts
(draft forward, verify forward, KV writes) are baked into traces whose input buffers
live at fixed addresses — that is the whole trick. The `QWEN36_DFLASH_DEVICE_LOOP=1`
path (`dflash_devloop_run.py:732`) goes further: the accept/commit math itself moves
on-card as "glue" traces (`_glue_a/_glue_b/_glue_c`, `dflash_devloop_run.py:294-499`),
leaving one upload and one readback per vLLM step.

The slot machinery is a **bucket** system (`Bucket`, `dflash2_serving.py:237`): each
bucket is a `(B_cfg, T_cfg)` verify geometry with its own verify trace, GDN spec cfg,
and drafter layout; `BucketPlanner.decide` (`dflash2_serving.py:142-181`) picks the
geometry from the live-slot count, with hysteresis so the bucket never flips twice in
one step (`dflash2_serving.py:49-53`). Slots that are live but not stepping *hold*:
their rows replay with `position = -1`, which the decode SDPA and the KV write both
skip (`_hold_inputs`, `dflash2_serving.py:946`; the `-1` skip is a decode-SDPA
contract, `attention/tp.py:953+`).

The 96-97%-under-load claim (`README.md:40-41`) is then structural: between the one
host touch (stage + accept, microseconds of Python) the card replays three traces with
no host round-trip. Our decode today is host-touched per layer per step
(`src/vt/tenstorrent/tenstorrent_backend.cpp`, `flint_tt.cpp` eager op-by-op dispatch);
the bundle's loop is host-touched per *step*. That is the property to copy, and it is
architectural (fixed-address buffers + traces), not a kernel-speed difference.

## 3. Mesh execution (Q3): 1x4 FABRIC_1D

- Weights: attention is column-parallel on q/k/v/gate and row-parallel on `o`
  (`attention/tp.py:88-97`, `tp.py:115`); the GDN in-proj is likewise sharded
  (`gdn/tp.py:191+`). Every row-parallel output ends in a `tt_all_reduce` +
  `tt_all_gather` pair on `cluster_axis=0` (`dflash2_tp.py:370-388`,
  `v1/ccl.py`).
- Activations: replicated. The drafter's rows are replicated full-width on every die
  (`dflash2_tp.py:119+` docstring: "Activations stay REPLICATED full-width"), so the
  draft forward is compute-identical on all 4 dies and needs no activation sharding
  logic — only the weight shards and the post-matmul collectives.
- The drafter's context KV is per-die replicated DRAM (`alloc_store`,
  `dflash2_tp.py:532-598`), *not* mesh-sharded: each die holds the whole drafter KV.
  Only the target's KV is paged per-slot on device.
- Fabric: the boundary ops use `FABRIC_1D` ring multicast for the all-reduce
  (`fabric_boundary/ops.py:51-141`: the sender packs the partial sum flat, multicasts
  `fwd_hops` forward and `bwd_hops` backward over the ring, and the compute kernel sums
  the ring slots in slot order — `fb_compute.cpp:12-30`).

The mesh multiplier is real but not free: the drafter is replicated (4x the weight
bytes per die), the target's attention KV is paged on device per-slot, and every
row-parallel matmul pays a collective. The bundle chose replication for the drafter
because 5 layers x ~1.8B params fits; a bigger drafter would have to shard.

## 4. Portability to our models (Q4)

What the pattern needs from a model, in order of how hard it is to get:

1. **Residual taps.** The drafter consumes the target's post-layer residual
   (`fc(hidden_norm(taps))`, `dflash2.py:532`). Any model with a residual stream can
   produce them; the tap copies are baked into the verify trace at fixed addresses
   (`model.py:453-457`, `model.py:1780-1800`).
2. **A trained drafter checkpoint.** This is the real lift: the bundle's drafter is a
   5-layer transformer trained against Qwen3.8-27B's residual stream
   (`incoai/Qwen3.8-27B-DFlash2`, `dflash2.py:39`). For kolibri1 or flint we would
   have to train one — the pattern transfers, the weights do not.
3. **A verify trace with fixed-address I/O.** Our TT backend already has capture
   (`tenstorrent_capture.cpp`); what is missing is the *contract* that the trace's
   input buffers live at fixed addresses so a replay reads only staged tensors.
4. **Bucket geometries.** `(B, T)` pairs with hysteresis (`BucketPlanner`,
   `dflash2_serving.py:142`). Our P150 decode has no bucket concept today; adding one
   is a serving-layer change, not a kernel change.

Per model: **kolibri1** is the closest target (single residual stream, GDN-free, our
TT decode already captured); **flint** would need the tap plumbing added to
`flint_tt.cpp`'s forward; the MoE models need the union-expert decode path
(`moe_decode/union.py`) before the drafter's block forward is cheap enough to be
worth it.

## 5. The adoption sketch (Q5)

Staged, smallest-first, each stage independently mergeable:

- **Stage 0 — the contract.** Fix the trace I/O shape: fixed-address input buffers,
  one stage call per step, no allocation between capture and replay. This is a
  prerequisite for every later stage and is testable on its own (capture, replay
  twice, assert byte-identical outputs).
- **Stage 1 — trace the existing decode.** Capture our current per-step forward as a
  trace with the stage contract. No drafter yet. This alone should show the
  host-touch-per-step property on our numbers.
- **Stage 2 — a drafter-shaped hole.** Add the tap copies to the verify trace and a
  stub drafter that returns the greedy argmax (accept rate 0, correctness still
  exact). This proves the loop can absorb a drafter without changing outputs.
- **Stage 3 — a real drafter.** Train or adapt a small drafter against the target's
  residual stream. This is the expensive stage and is where the bundle's ~500x
  advantage actually lives — the loop is the cheap part.
- **Stage 4 — buckets.** Add `(B, T)` geometries with hysteresis once multi-user
  decode matters.

Risks, named: (a) fixed-address buffers are an aliasing hazard — a stale handle
freed between capture and replay corrupts the trace silently (the bundle's
`_set_anchor` comment at `spec_decode.py:976-979` names exactly this); (b) the
drafter's accept-rate is model-specific and unmeasured for ours — a drafter below
~1.5 accepted drafts/iteration loses to plain decode on the bundle's own numbers
(`model_config.py:43`: DFlash v1 K=11 measured 107.8 tok/s vs native MTP 80.9);
(c) bucket hysteresis interacts with slot lifecycle — the bundle's `BucketPlanner`
cooldown rules (`dflash2_serving.py:49-53`) exist because a bucket flip per step is
a performance cliff.

## 6. License and attribution

Apache-2.0 end to end: the bundle README declares it (`README.md:3`), every fetched
source carries an SPDX header, and the drafter checkpoint's card does the same. A
future adaptation owes: (a) SPDX header retention in any copied kernel or op file;
(b) attribution in the file that copies (not a blanket NOTICE entry) — the sources
are per-file licensed; (c) the PROVENANCE chain recorded in whatever row spec adopts
the pattern, so the next reader can find the pinned revision. The bundle's own
provenance discipline (`PROVENANCE.json`: oracle repo, revisions, wheel sha256s, the
`code_sha256` of the served code) is worth copying regardless of whether we adopt the
pattern — it is what made this study citable at a pinned sha.

## 7. What we did not verify

- No weight downloads: the drafter's checkpoint contents are known only from its
  config/README metadata and its use sites in the fetched code.
- No device run: every number in the bundle's README is quoted as a claim with a
  source line, never as a measurement we reproduced. The row's gate is the written
  note, not a benchmark.
- The `QWEN36_DFLASH_DEVICE_LOOP=check` mode's glue-trace comparison
  (`dflash_devloop_run.py:813-838`) is read from source, not exercised.
