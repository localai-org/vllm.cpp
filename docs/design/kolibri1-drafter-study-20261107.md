# The DFlash2 adoption sketch for kolibri1 on the P150 — a design study

- Row: `KOLIBRI1-DRAFTER-STUDY` (study row; the deliverable is this note, not code)
- Parent: `TT-DRAFTER-STUDY`, [docs/design/tt-drafter-study-20261103.md](tt-drafter-study-20261103.md)
  (on `mine/row/TT-DRAFTER-STUDY`). That study is the mechanism source: every
  DFlash2 claim below cites it, and it in turn cites the fetched bundle
  `mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4`
  (Apache-2.0, scratch copy at `/tmp/df2-bundle/`, provenance verified against
  `PROVENANCE.json` at this row's pinned revisions — oracle revision
  `28add9a1fbd0`, drafter checkpoint `dedf8df6…` recorded there at :73).
- License/attribution: Apache-2.0 end to end. This note inherits the parent
  study's attribution; any future adaptation owes the SPDX-header, per-file
  attribution, and provenance-recording duties that study's §6 names.
- No device run: this row is arithmetic + structure. Every measured number is
  quoted from a recorded bench-evidence file with its citation.
- A premise correction up front, because two sections depend on it: **the spec
  premise "kolibri1's TT decode is already captured/replayed with one host
  touch per step" is falsified by the tree.** kolibri1's TT decode is an eager,
  host-serialized loop; capture wiring never landed, and the gap analysis
  records it as a campaign, not a task (`docs/gap-kolibri1-tt-captured-decode.md`,
  "kolibri1 never captures", "Why full captured decode is NOT feasible here").
  Sections 2 and 5 are written against that corrected picture.

## 1. kolibri1's TT decode structure today

**The model shape** (the loader's own header): 50 backbone layers, EVERY layer
MoE — 384 routed experts, top-6, plus one ungated shared expert
(`src/vllm/model_executor/models/kolibri1_weights.h:7-9`); hidden 2560, vocab
128000; hybrid attention — 40 sliding-window layers at window 513 and 10
full-attention RNoPE layers, with per-head q/k RMS norms and a two-group KV
(`kolibri1_tt_forward.h:17-20`). The released checkpoint is fp8
block-quantized, `weight_block_size [128,128]`, with the bf16 router and f32
bias outside the conversion (`kolibri1_weights.h:13-22`). The lm_head is UNTIED
(`kolibri1_tt_forward.cpp:1385-1387`), and the residual stream is bf16 `[t,2560]`
through the whole forward (`kolibri1_tt_forward.cpp:1200-1211`).

**The forward is a hand-rolled eager loop.**
`ForwardKolibri1TTResidentForward`
(`kolibri1_tt_forward.cpp:1144-1395`) runs, per step: embedding into a held
`DBuf` (`:1200-1209`), then per layer (`:1220`): fused add-RMSNorm
(`:1261-1267`), attention (`AttentionBlock`, `:362-499`, called at `:1306`),
post-attn norm (`:1310-1313`), the post_attention_layernorm add-norm
(`:1318-1328`), the MoE block (`MoeBlock`, `:528-1016`, called at `:1334`),
post-ffn norm (`:1337-1339`), then the final add-norm (`:1355-1366`) and the
untied lm_head matmul into f32 logits (`:1385-1387`). Every step allocates
fresh `DBuf`s through the whole loop — there are no persistent per-step device
buffers and no `Replay()` (`docs/gap-kolibri1-tt-captured-decode.md`, missing
item 1).

**The host-touch boundary is per-LAYER, not per-step.** The router downloads
its full `[t,384]` f32 logits to host each layer (`kolibri1_tt_forward.cpp:563-564`),
runs the sigmoid-logit-add top-6 on host (`:582`,
`kolibri1_shared.h::SigmoidLogitAddRouting`), and the routed experts then
stream from host: dispatch plan (`:673-677`), fetch list (`:678-680`), a
per-expert H2D stage with a full-slot D2H readback pivot verified byte-exact
(`:685-702`), a per-slot bf16 dequant memo (`:669`, evict/emplace at
`:1054-1067`). The bias is a static weight, cached host-side once per device
buffer to remove 50 blocking round-trips per step (`:565-581`). Counted plainly,
that is two-plus blocking host round-trips per layer, ~100+ per step — the gap
analysis names exactly this as what forbids trace capture (`docs/gap-kolibri1-tt-captured-decode.md`,
missing items 3-4). A device-router composite exists and is gate-verified but
measured SLOWER and is env-gated off (`kolibri1_tt_forward.cpp:540-548`).

**The slot machinery.** The device half of the streaming MoE holds an fp8
packed expert-slot pool on the card: host-side slot-pool plan
(`kolibri1_tt_stream.cpp:60-145`), the fetch list executor
(`kolibri1_tt_stream.cpp:147-207`), the per-step stream-bound guard with a LOUD
refusal past the B1 per-token ceiling (`kolibri1_tt_stream.cpp:209-217`),
and the slot shadow — the readback pivot's reference side
(`kolibri1_tt_stream.cpp:219-293`). The LRU policy lives host-side
(`kolibri1_tt.cpp:329-426`: `Touch` `:367`, `SlotFor` `:398`, `ResidentCount`
`:407`), with the same-step residents refreshed before the fetch loop so an
eviction never claims an expert this step selected (`kolibri1_tt_forward.cpp:703-711`).

**The capture status, one sentence:** the qwen3.5 captured-decode template
exists and is proven on this card (~12 ms wall / 0.3 ms replay per served
step), but kolibri1 implements none of its hooks — no decode-graph driver, no
Warm\* hooks, per-layer D2H mid-loop, host-built data-dependent expert fetch —
and the gap analysis prices the ordinary part at 3-5 device-days with ~zero
win while items 3/4 still serialize every layer
(`docs/gap-kolibri1-tt-captured-decode.md`, items 1-5 and the honest verdict).

**The measured baseline.** The P150 anchoring session resolves the 6x
discrepancy and lands the current head at **983.5 ms/step, 1.017 tok/s**
(`docs/bench-evidence/tt-anchoring-20261030/README.md:44`, cell head-perf;
governor swept, contamination marked). The best recorded P150 vintage is
548.5 ms/step / 1.823 tok/s at `3ce14f35c` (same record, `:47`), with a residual
1.77x unexplained gap owed (`:74-77`). The device golden-argmax gate is RED at
BOTH code eras, root-caused to substrate-level numeric divergence in the device
execution path, not our code (`:79-101`) — recorded here because §5 leans on it.

## 2. The drafter fit

DFlash2 needs four things from a target (parent study §4): residual taps, a
trained drafter, a verify trace with fixed-address I/O, and bucket geometries
(tt-drafter-study-20261103.md, "What the pattern needs from a model").

**What kolibri1 provides already.**

- **Residual taps exist.** The drafter consumes the target's post-layer
  residual (`fc(hidden_norm(taps))`, per the parent study §1 citing
  `dflash2.py:532`). kolibri1's forward materializes a bf16 `[t,2560]` residual
  stream with an explicit post-layer value at every layer boundary —
  `res`/`hidden` at `kolibri1_tt_forward.cpp:1211`, updated by the fused
  add-norms at `:1261-1267`, `:1318-1328`, read at the final norm `:1355-1366`.
  Tap copies would be baked into the verify trace at fixed addresses, exactly
  as the parent study describes (§4, item 1).
- **A verify pass** is the target's own forward. kolibri1 has one, and the
  acceptance math needs only its argmax — greedy verify is token-exact because
  the accept rule IS the target's argmax (parent study §1, item 3). On-device
  argmax sampling already landed on this seam (`kolibri1_tt_forward.h:23-25`).
- **MoE is not a blocker structurally**: the DFlash2 target (qwen3.8-27B) is
  also MoE, and the drafter attends only residual taps plus its own context KV,
  never the expert tiers.

**What must be built — and this is the section's load-bearing finding:**

1. **The captured decode itself.** The parent study names fixed-address trace
   I/O as prerequisite stage 0, and kolibri1 does not have ANY of it — the
   drafter pattern's whole trick (stage → replay → accept, host-touched per
   step) presupposes a captured verify. kolibri1's eager loop cannot host a
   drafter: the verify trace does not exist, and the per-layer router readback
   plus host-driven expert streaming is uncapturable by construction
   (`docs/gap-kolibri1-tt-captured-decode.md`, items 3-4). Stage 1 below is
   therefore the capture campaign the gap analysis already priced, not a
   small trace fix.
2. **The drafter's own paged context KV** and its extend path (parent study §1
   item 4). kolibri1's TT KV today is the checkpoint's per-layer paged cache
   (`kolibri1_tt_forward.cpp:1224-1236`); the drafter's KV is a second,
   drafter-private structure the P150 arm does not build today.
3. **The bucket system** — `(B, T)` verify geometries with hold semantics and
   hysteresis (parent study §2, `BucketPlanner`). kolibri1's P150 arm has no
   slot/geometry concept at serving level; this is serving-layer work, last.

## 3. The drafter checkpoint plan

**The qwen3.8 drafter does not transfer.** Different vocab (128000 vs the
drafter's), different hidden size, and the drafter is trained against the
TARGET's residual stream — its weights are a distillation of qwen3.8-27B's
specific layer outputs (parent study §4 item 2: "the pattern transfers, the
weights do not"). A kolibri1 drafter must be TRAINED.

**Size.** A DFlash2-shaped drafter is 5 layers consuming the target's taps
(parent study §1 item 1). At kolibri1's hidden 2560, a per-layer transformer
block is roughly: attention 4 projections ≈ 4·2560² ≈ 26.2M params; SwiGLU MLP
at a qwen-like ~3-4x hidden intermediate ≈ 3·2560·(7680-10240) ≈ 59-79M.
That is ≈ 85-105M params per layer, so a 5-layer drafter lands at
**≈ 0.42-0.52B parameters, ≈ 0.85-1.05 GB bf16** — comfortably resident on the
P150 alongside the target (the bundle's own drafter is described as
"5 layers x ~1.8B-class" replicated per die, parent study §3, so kolibri1's is
~4x smaller than the proven one). The context KV for a 2K-token drafter context
adds ~5 layers x 2K x 8 KV heads-ish x 128 head-dim x 2 x 2 bytes — tens of MB,
not a budget problem.

**Training data + compute.** The training objective is distillation against
kolibri1's own residual taps and argmax (draft-block reconstruction under the
teacher's frozen forward). Order-of-magnitude: 6·N·D FLOPs with N ≈ 5e8 and
D ≈ 1e11 tokens gives ≈ 3e20 FLOPs — on an 8-GPU node at ~4e14 effective bf16
FLOP/s that is ≈ 8-9 days of dedicated training, plus the data pipeline
(kolibri1-domain prefill, tap capture at fp8-checkpoint fidelity). This is a
multi-week, multi-GPU row of its own, and it is UNMEASURED need: nothing in
the tree says a 0.5B drafter on kolibri1's residual stream reaches any
particular acceptance rate.

**The acceptance target.** The parent study's break-even floor is ~1.5 accepted
drafts per iteration, measured on the bundle's own numbers
(tt-drafter-study-20261103.md §5 risk (b), citing `model_config.py:43`). §5
below shows kolibri1's floor is materially HIGHER than that in the current
regime — the ~1.5 floor presumes a verify pass whose cost does not scale with
the block length, which kolibri1's does not have today.

**The honest risk register.**

- **Accept rate is unmeasurable before the drafter exists.** Training is the
  expensive stage and the acceptance number that justifies it only arrives
  after it (parent study §5 stage 3: "the expensive stage... where the bundle's
  ~500x advantage actually lives").
- **Below break-even loses to plain decode**, by construction: every accepted
  token then costs verify-plus-draft for less than one plain step.
- **The greedy-accept contract currently has no stable device reference.** The
  device golden-argmax gate is red at both eras on substrate divergence
  (`tt-anchoring-20261030/README.md:79-101`); a drafter whose accept rule is
  "the device's own argmax" inherits that instability — accepted tokens would
  be exactly the device chain, which is exactly what the substrate flips.
- **Training cost vs payoff**: §5 estimates the ceiling of the win at kolibri1's
  scale well below what a multi-week training campaign plus a capture campaign
  pays for elsewhere on the same card.

## 4. The staged adoption plan (adapted from the parent study's five stages)

Smallest first, each stage independently mergeable, each with its gate. The
parent study's stage 0 here EXPANDS, because kolibri1 lacks the whole captured
substrate the study assumed:

- **Stage 0 — the stage contract, scoped to what exists.** Fix trace I/O to a
  contract (fixed-address input buffers, one stage call per step, no allocation
  between capture and replay) against the qwen3.5 template, testable by
  capture-replay-replay byte-identity (parent study §5 stage 0). Gate: the
  byte-identity test green on the P150 for a dense model.
- **Stage 1 — trace kolibri1's EXISTING decode.** This is the gap analysis's
  campaign: the decode-graph driver, the Warm\* hooks, the block_size-32 cache
  fix, and — the hard part — the router readback and host-driven expert
  streaming made capturable (device routing or a resident/union-expert arm;
  `docs/gap-kolibri1-tt-captured-decode.md`, "Why full captured decode is NOT
  feasible here"). Gate: kolibri1's decode replays at host-touch-per-step and
  the 141/145 golden gate goes green on the captured chain — the substrate
  argmax gate currently blocks this stage's gate.
- **Stage 2 — the drafter-shaped hole.** Add the tap copies to the captured
  verify trace and a stub drafter returning the greedy argmax (accept rate 0,
  correctness still exact — parent study §5 stage 2). Gate: identical token
  chains with the stub drafter in and out of the loop.
- **Stage 3 — the trained drafter.** Train the ~0.5B drafter (§3), wire draft +
  extend as traces. Gate: measured accept rate; GO only above the §5 floor,
  computed on the card, not modeled.
- **Stage 4 — buckets.** `(B, T)` geometries with hysteresis once multi-user
  decode matters (parent study §5 stage 4). Gate: no bucket flip within one
  step under a churn workload.

## 5. The decision: NO-GO, with the basis and the flip conditions

**Recommendation: NO-GO for a drafter on kolibri1/P150 today.** Four
independent reasons, any one of which is disqualifying:

1. **The verify pass does not amortize, and that is fatal at kolibri1's
   operating point.** Speculative decode wins when one verify forward over
   [pending + K drafts] costs ≈ one plain step. The qwen3.8-27B bundle has
   that: a captured, batch-amortized verify on a 4-die mesh (parent study §2).
   kolibri1's P150 decode is an eager, host-serialized, batch-1 GEMV path —
   983.5 ms/step with ~100+ blocking host round-trips (`tt-anchoring-20261030/README.md:44`;
   `docs/gap-kolibri1-tt-captured-decode.md` items 3-4). A GEMV per token does
   not amortize a block: verifying K+1 rows costs ≈ (K+1)× a 1-row step, so
   the speedup S ≈ τ/(K+1) < 1 for every K ≥ τ and every achievable τ. Even
   granting a generous K=4 block and τ=1.5 accepted drafts, S ≈ 0.3 — a 3x
   SLOWDOWN. The break-even floor at kolibri1's current operating point is not
   ~1.5; it is unreachable.
2. **The loop precondition does not exist.** The drafter pattern presupposes
   the stage-replay-accept captured loop; kolibri1 never captures, and making
   the MoE arm capturable is the multi-row campaign the gap analysis already
   recorded (§1 above). The drafter's marginal integration cost is therefore
   NOT smallest here — it is the sum of that campaign plus a drafter campaign.
3. **The drafter checkpoint must be trained**, ≈ 0.5B params, multi-week
   compute (§3), against a target whose device argmax — the accept rule's
   entire semantics — is currently substrate-unstable on this card
   (`tt-anchoring-20261030/README.md:79-101`).
4. **The payoff ceiling is small where the effort is large.** Grant the
   optimistic future — captured amortized verify (stage 1 landed), trained
   drafter at the bundle's τ=1.5 floor: S ≈ 1.5/1.15 ≈ 1.3, taking the head
   from 1.02 to ≈ 1.33 tok/s. For that ~0.3 tok/s, the cost is a capture
   campaign, a `MoeRouterTopK`/resident-MoE campaign, and a multi-week
   training run. The same effort spent on the recorded gap — the 1.77x
   unexplained head-vs-1021 regression (`tt-anchoring-20261030/README.md:74-77`)
   — has a larger measured ceiling (1.02 → 1.82 tok/s is already demonstrated
   by the same card at `3ce14f35c`), and the CPU GGUF arm's llama.cpp gap
   (4.97 vs 17.98 tok/s, `docs/USAGE.md:1017-1031`) dwarfs both.

**What would flip this to GO**, each a named, measurable precondition:

- Stage 1 lands: kolibri1's decode is a captured, batch-amortized, host-touched
  per-step replay whose golden gate is green — at which point a block verify
  costs ≈ one step and the ordinary ~1.5-token break-even floor applies.
- A trained (or adapted) drafter measures τ ≥ 2 on the card against the
  captured chain — measured, not modeled, per the parent study's rule that no
  number is ours until the card says it.
- The substrate numeric divergence that reds the device golden gate is
  resolved, so greedy acceptance has a stable device reference.

Until then the honest record is: **kolibri1's decode is already on-card at
1.02 tok/s, its step is host-serialized in a way a drafter cannot repair, and
the drafter's training cost exceeds the projected win at kolibri1's 2.8B-active
scale.** The adoption row that inherits this study should open on stage 1's
capture campaign only if the capture work is wanted for its own sake (it is
recorded as owed under `BACKEND-TENSTORRENT`), never as the first move of a
drafter plan.
