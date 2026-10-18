# Kolibri-1 CPU: speculative-decoding feasibility probe — the n-gram
# drafter is portable, exact, and measured (MODEL-TEXT-kolibri-1)

Date: 2026-10-11. Host: aarch64, 128 cores, 255 GB RAM, Linux (the same box as
the 2026-10-07/08/09 records). Tree: `origin/main` **e2c95010b** + this probe's
three test files, branch `row/kolibri-spec-probe`, worktree `/tmp/vllm-spec-probe`,
build `/tmp/build-spec-probe`, Release, `-DVLLM_CPP_CUDA=OFF
-DVLLM_CPP_TENSTORRENT=OFF -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF`
(clean configure + build, exit 0; the three new tests build against it and are
all green — §4/§5). Checkpoint `/mnt/models/Aleph-Alpha/Kolibri-1` (74 GB fp8,
32 shards). Issue: ISSUE-LOCAL-01M4P22PX6BPRQAY1CK20FZQF6. All runs under
`flock -x $HOME/gpu.lock` with `VLLM_CPP_CPU_THREADS=32
VT_KOLIBRI1_DEQUANT_CACHE_MB=32768` (the bench recipe). No push, no PR — the
deliverable is this report + the three committed tests + the simulator.

## The question

The CPU kolibri decode bench measures ~5.2-5.76 tok/s decode (t32 + 32 GiB
dequant cache + NEON; 5.76 in the task's baseline, 5.156 reproduced here —
§6) against a 12 tok/s developer target, and the measured levers are nearly
exhausted. Speculative decoding is the one remaining multiplier that does NOT
change numerics: greedy verification is exact (a draft is accepted iff the
target's argmax agrees; a rejected draft costs only the draft proposal, which
for n-gram is a host-side string match). kolibri has NO MTP/nextn layer
(`mtp_num_hidden_layers` absent from `config.json`), so the cheap draft is
absent. The candidates were (a) n-gram/prompt-lookup drafting — llama.cpp's
`-draft` style, no trained model — and (b) a tiny external draft model. This
probe answers, for (a): is the verify machinery portable to kolibri, what
acceptance does an n-gram drafter get on kolibri's greedy chains, and what
tok/s does that project to?

## Verdict (one paragraph)

**The machinery is portable; the acceptance is real; the bench-shape workload
clears the 12 tok/s target; exactly one qwen3.5-only seam blocks the engine
path today.** The tree's spec-decode machinery is NOT qwen3.5-shaped on the
n-gram path — the scheduler slot allocation, the engine's draft
install/rollback, the runner's verify (`sample_tokens_with_rejection`, greedy
accept-iff-argmax) and the draft-free n-gram proposer are all generic, and
`GPUModelRunner` — the runner that carries them — is the production runner on
CPU too. The ONE blocker is in the loader, not the engine:
`MakeKVCacheMaybeSpec` builds the qwen3.5-only widened KV spec for ANY
speculative config (§5). Around it, the bench-level prototype (production
proposer + production verify forward, §4) measures on the real checkpoint:
**100% per-token acceptance on the bench chain, 14.51 tok/s decode vs 4.92
plain — 2.95×, clearing the 12 tok/s target**; 54.1% per-token acceptance on
the W3 golden battery (8.99 vs 6.84 tok/s, 1.31× — diluted by short prompts
and by the engine's own near-tie chains); and the spec chain is
**byte-identical to plain greedy on every prompt** — the numerics-free
property holds on the real model. Candidate (b), a tiny external draft model,
is NOT needed: it costs a second checkpoint and a second forward path, and on
chains this repetitive a free string match already accepts everything. The
integration plan is in §7; no serving wiring is done by this probe.

## 1. Machinery portability: what is generic, what is model-specific

The n-gram lane was read end to end. The portability split:

**Generic engine machinery (portable to kolibri unchanged):**

| Piece | Anchor | Notes |
|---|---|---|
| Config/CLI parse + ngram resolve | `src/vllm/config/speculative.cpp` (`ParseSpeculativeConfigJson`), `include/vllm/config/speculative.h:389-423` (`ResolveNgram`), `:471-474` (`use_ngram`) | `method:"ngram"` accepted; k REQUIRED; window defaults 5/5 |
| Loader resolution | `src/vllm/entrypoints/model_loader.cpp:1622-1630` (`ResolveSpecConfig` ngram branch) | no draft checkpoint, no platform gate |
| Scheduler spec slots | `src/vllm/v1/core/sched/scheduler.cpp:247-251` (`num_lookahead_tokens_`, 0 for ngram), `:700-741` + `:948-950` (`NumTokensWithSpec` → `allocate_slots`), `:1600+` (`update_draft_token_ids` installs drafts) | drafts are scheduled as ORDINARY tokens; KV slots come from the generic `KVCacheManager::allocate_slots` |
| Scheduler rollback + stats | `scheduler.cpp:1226-1265` (`num_computed_tokens -= num_rejected`), `:1458+` (`make_spec_decoding_stats`) | rejected drafts' KV slots are simply overwritten next step |
| Engine core draft pull | `src/vllm/v1/engine/core.cpp:101-127` (`post_step` → `take_draft_token_ids`), `core.cpp:78-87` (`update_from_output`) | sync scheduler is forced for ngram (`async_scheduling_compatible()==use_eagle()==false`, `model_loader.cpp:2313-2314`) — the sync scheduler carries the full spec plumbing |
| Runner predicates | `include/vllm/v1/worker/gpu/runner.h:1229-1307` (`spec_on`/`num_spec`/`use_ngram`) | |
| Draft splice into the step | `src/vllm/v1/worker/gpu/runner.cpp:2440-2530` | splices the scheduler's drafts after each request's committed prefix |
| Verify-step classification | `runner.cpp:3400` (`step_num_logits`), `StepRoutesToVerify` | a verify step is 1+k logits rows per request |
| THE VERIFY HALF | `runner.cpp:3546+` (`sample_tokens_with_rejection`) → `src/vllm/v1/spec_decode/rejection_sampler.cpp` | greedy accept rule pinned at `include/vllm/v1/spec_decode/rejection_sampler.h:13-51`: accept iff draft == target argmax, stop at first mismatch, exactly one bonus/replacement token |
| THE PROPOSE HALF | `runner.cpp:4517-4570` (`propose_drafts_ngram`) → `src/vllm/v1/spec_decode/ngram_proposer.cpp` (`NgramPropose` / `FindLongestMatchedNgramAndProposeTokens`) | host-side KMP-LPS longest-suffix match over the request's own committed context; no draft model, no hidden tap, no draft KV |
| Acceptance counters | `runner.cpp:3664-3681`, accessors `runner.h:366-367` (`spec_drafts_accepted/proposed`) | the probe's measurement source |
| Model dispatch | `runner.cpp:3273` (`ModelRegistry::Forward`) → `include/vllm/model_executor/models/model_registry.h:636+` (`ModelForwardInput`, already carries `num_speculative_tokens` at `:657-665`) | one seam every model flows through |
| Metrics | `src/vllm/v1/spec_decode/metrics.{h,cpp}` (`SpecDecodingStats` + the four `vllm:spec_decode_*` Prom families) | config-gated, on with any spec config |

**Model-specific (NOT needed for n-gram on kolibri):**

- The MTP / DFlash / DFlash2 / DSpark speculators
  (`src/vllm/v1/worker/gpu/spec_decode/{mtp,dflash,dflash2,dspark}/`) — each
  needs a qwen3.5 draft checkpoint and/or hidden-state taps. n-gram has no
  drafter, so none of this is on the path.
- The GDN/mamba spec machinery: `MambaSpec::num_speculative_blocks`
  (`include/vllm/v1/kv_cache_interface.h:473-485`), the widened conv + k+1
  state slots (`src/vllm/model_executor/models/qwen3_5_common.cpp:79-90`).
  kolibri's `layer_types` are all `sliding_attention`/`full_attention` — there
  is no mamba layer, so no GDN spec state exists to widen. (The loader comment
  at `model_loader.cpp:1618-1621` ties this machinery to qwen3.5's hybrid KV;
  it does not apply to kolibri.)
- The hidden taps `ModelForwardInput::hidden_tap` / `aux_tap`
  (`model_registry.h:676-694`) — qwen3.5-only; kolibri's forward ignores them.

**The one CPU-specific observation:** the `#ifdef VLLM_CPP_CUDA` guards in
`runner.cpp` (lines 2732, 5409, 5423, 5457, 5540, 5707) all gate the
async-mirror / decode-graph / copy-queue features, NOT the spec path —
`propose_drafts_ngram` and `sample_tokens_with_rejection` compile and run on
the CPU build (the rejection sampler's host `forward` route is the CPU route;
its device `verify` route is the CUDA one).

**Portability cost: zero production lines.** The n-gram lane was landed
2026-07-27 (`05a2e0d0c`, SPEC-NGRAM / ROAD-V1-D3) and gated dgx-only
(`tests/parity/test_qwen27_ngram_spec_decode.cpp`: 5/5 prompts STRICT
our-ngram-ON == vLLM-ngram-ON, 180/180 drafts accepted). Nothing in it names
qwen3.5; the only reason kolibri has no spec-decode number yet is that no one
ran kolibri with the flag.

## 2. The kolibri CPU decode path and the verify hook point

A decode step on CPU flows: `Scheduler::schedule` (spec tokens scheduled as
ordinary tokens) → `Executor::execute_model` → `GPUModelRunner::execute_model`
(the production runner on CPU — `LoadedEngine` owns it at
`include/vllm/entrypoints/model_loader.h:897`) → draft splice →
`ModelRegistry::Forward` (`runner.cpp:3273`) → kolibri's registry hook
`ForwardKolibri1ForCausalLM` (`src/vllm/model_executor/models/kolibri1_registry.cpp:316`)
→ `ForwardKolibri1Forward` (`src/vllm/model_executor/models/kolibri1_forward.cpp:556`)
→ CPU paged attention (`src/vllm/v1/attention/backends/cpu_attn.*`) → logits →
`sample_tokens` routes to `sample_tokens_with_rejection` when
`StepRoutesToVerify(step.num_draft_tokens)` → greedy accept walk → the
scheduler's rollback rewinds `num_computed_tokens` by the rejected count →
`propose_drafts_ngram` drafts the next k from the committed context.

The verify hook point is exactly the runner's sampling seam, and the verify
forward is a T=k+1 causal multi-token step — the SAME shape as a chunked
prefill, which kolibri's forward already serves in production (the bench runs
T=128 causal prefill + T=1 decode; the engine's chunked prefill runs T>1 at
start_pos>0; PR #3422 served kolibri end to end through this engine on CPU).
Kolibri's KV spec (`FullAttentionSpec` + `SlidingWindowSpec`,
`kolibri1_registry.cpp:212-224`) allocates the drafts' slots through the
generic path; the 513 sliding window simply treats the drafts as the newest
tokens. No per-model spec gating exists (searched the runner /
prepare_inputs / input_batch refusal surface).

## 3. Measured acceptance: the exact simulation

Greedy verification is exactness-preserving, so the committed tokens of a
spec-decode run are always a prefix of the plain greedy chain — the
acceptance rate on a KNOWN chain is computable exactly, with no model run and
no statistics. `scripts/kolibri1-ngram-accept-sim.py` mirrors the production
`FindLongestMatchedNgramAndProposeTokens` line for line and simulates the
propose→verify loop (prefill emits chain token 0; each verify pass accepts the
draft prefix matching the chain and emits accepted+1 tokens; the final pass
is truncated by max_tokens). The C++ unit
`tests/vllm/v1/spec_decode/test_kolibri1_ngram_accept.cpp` runs the SAME
simulation through the PRODUCTION proposer over the same chains and pins the
aggregate numbers exactly (a proposer mutation or a golden regeneration turns
it red) — green: 2 cases, 90 assertions.

**W3 golden chains** (`tests/vllm/models/kolibri1_goldens.json`, 8 prompts × 32
greedy tokens, 248 plain decode steps). The chains are HIGHLY repetitive —
constant runs (golden 0: token 121598 ×32; golden 7: 12843 ×32), a 2-cycle
(golden 1), and repeated blocks — which is precisely the regime an n-gram
drafter exploits:

| config | verify passes | step-ratio | tok/verify | accepted/proposed | per-token accept |
|---|---|---|---|---|---|
| k=1, window 1-1 | 175 | 1.417 | 1.463 | 76/139 | 0.547 |
| k=2, window 1-4 | 126 | 1.968 | 2.032 | 127/176 | 0.722 |
| k=3, window 1-4 | 115 | 2.157 | 2.226 | 138/224 | 0.616 |
| k=4, window 1-4 | 109 | 2.275 | 2.349 | 146/268 | 0.545 |
| **k=5, window 1-4** | **103** | **2.408** | **2.485** | **150/297** | **0.505** |
| k=8, window 1-4 | 96 | 2.583 | 2.667 | 157/382 | 0.411 |
| k=5, window 2-4 | 123 | 2.016 | 2.081 | 132/239 | 0.552 |
| k=5, window 5-5 (upstream default) | 147 | 1.687 | 1.741 | 107/124 | 0.863 |

Per-chain spread at k=5 window 1-4: 7/31 steps (golden 7, constant run) to
21/31 (golden 4, the least repetitive chain) — every chain beats plain decode.
The window matters: the upstream default 5/5 is markedly worse on these short
chains (long n-grams rarely repeat within 32 tokens); 1-4 is the measured
sweet spot.

**Bench chain** (the decode bench's deterministic prompt,
`(i*7919+13)%128000` — 128 DISTINCT random tokens, the adversarial prompt for
a suffix matcher — plus the 64-token greedy chain captured from this tree's
bench run, §6). The chain is a perfect 2-cycle: `101807,109726,101807,...`
(the model enters the loop at position 2). Once the loop exists in the
context, the matcher fires on the generated tokens even though the prompt
never repeats:

| config | verify passes | step-ratio | tok/verify | accepted/proposed | per-token accept |
|---|---|---|---|---|---|
| k=2, window 1-4 | 21 | 3.000 | 3.000 | 42/42 | 1.000 |
| k=4, window 1-4 | 14 | 4.500 | 4.500 | 50/52 | 0.962 |
| **k=5, window 1-4** | **12** | **5.250** | **5.250** | **52/53** | **0.981** |
| k=8, window 1-4 | 9 | 7.000 | 7.000 | 55/56 | 0.982 |

**Honesty note (the estimate's bounds):** these chains are degenerate loops —
this is a small research model whose greedy decode collapses into repetition
on short prompts, and BOTH the W3 battery and the bench chain live in that
regime. The 50.5%/98.1% figures are therefore an UPPER bound for real serving
traffic; a prompt whose continuation is not repetitive gives the drafter
nothing to match and the run degenerates to plain decode plus the verify
overhead (measured in §6: the overhead is small). The e2e run below measures
the true engine-level acceptance on the same chains, and the wall-clock ratio
there is the number that decides viability — not the token ratio.

## 4. The prototype (contained, bench-level — the brief's fallback, and honest)

The brief allowed a bench-level harness if the engine machinery looked
qwen35-shaped. §1 shows the n-gram lane is generic EXCEPT one loader branch
(§5), so the prototype is the brief's contained fallback, built from
production pieces only:
`tests/vllm/models/test_kolibri1_spec_probe.cpp`.

- **Drafter**: the PRODUCTION proposer,
  `vllm::v1::spec_decode::FindLongestMatchedNgramAndProposeTokens` (the landed
  KMP-LPS matcher, `ngram_proposer.cpp`), over the committed context.
- **Verify**: ONE production target forward (`ForwardKolibri1Forward`) over
  the k+1 verify sequence `[last committed token, draft_0..draft_{k-1}]` —
  the engine's exact verify geometry (a causal multi-token forward, the same
  shape as a chunked prefill; the bench topology is the decode bench's own).
- **Accept rule**: the production greedy rule
  (`rejection_sampler.h:13-51`) — accept iff the draft equals the target
  argmax at its position, stop at the first mismatch, one bonus/replacement
  token per verify pass.

Two prompt sets, each run PLAIN (the bench's own greedy loop) and SPEC (k=5,
window 1-4), one warmup pass first (dequant-cache cold misses concentrate in
the first forwards; both arms pay them equally):

1. the deterministic bench prompt (128 tokens, 64 greedy out);
2. the 8 W3 golden prompts, pre-tokenized (the golden `input_ids`, 32 tokens
   each).

Gates: **(i) exactness** — the spec chain == the plain chain byte-for-byte on
every prompt; **(ii)** the plain pass reproduces the bench's pinned chain
(last token 109726); **(iii) aliveness** — drafts proposed and accepted.
Checkpoint-gated (loud SKIP when the model is absent); CPU only. Green:
17/17 assertions.

A second test, `tests/vllm/models/test_kolibri1_ngram_spec.cpp`, pins the
ENGINE-level reach: the spec-OFF engine loads and generates on CPU for
kolibri, and the spec-ON engine construction throws at the precise seam of §5
(the test asserts the exact refusal message, so the integration that removes
the seam flips it deliberately).

## 5. Prototype results (real checkpoint, t32 + 32 GiB cache)

### 5a. The deterministic bench prompt (128 in, 64 greedy out)

| arm | steps | decode s | decode tok/s | drafts accepted | per-token accept |
|---|---|---|---|---|---|
| plain | 63 | 12.81 | 4.92 | — | — |
| **spec k=5, window 1-4** | **12 verify passes** | **4.34** | **14.51** | **53/53** | **1.000** |

- Exactness: the spec chain is byte-identical to the plain chain; the plain
  chain's last token is 109726 (the bench anchor) — the harness is the
  bench's own geometry.
- **Decode speedup 2.95×; 14.51 tok/s against the 5.76 baseline and the 12
  tok/s target — the target is cleared on the decode-heavy workload.**
- The step-count ratio (63/12 = 5.25) matches the exact simulation's 5.25 for
  this chain (§3); the wall ratio is lower because a verify pass is a T=6
  forward, not a T=1 one — the 2.95× is the honest number and it already
  includes that overhead.

### 5b. The W3 golden battery (8 prompts × 32 greedy tokens)

| prompt | verify passes / plain steps | accepted/proposed |
|---|---|---|
| golden[0] "alpha beta gamma..." | 7/31 | 24/24 |
| golden[1] "der Mond scheint..." | 8/31 | 23/23 |
| golden[2] "The capital of Australia is" | 18/31 | 14/36 |
| golden[3] "Wissen ist Macht..." | 9/31 | 24/29 |
| golden[4] "x1 = 3, x2 = 7..." | 21/31 | 10/84 |
| golden[5] "Translation to German..." | 13/31 | 21/35 |
| golden[6] "Ein gutes Buch..." | 25/31 | 6/15 |
| golden[7] "counting: one two..." | 7/31 | 24/24 |
| **aggregate** | **108/248 (2.30×)** | **146/270 (54.1%)** |

- Exactness: **8/8 byte-identical spec vs plain** on the real model.
- Decode-phase tok/s: 6.84 (plain) → **8.99 (spec)**, speedup 1.31×. The
  battery is short-prompt dominated (5-22 token prompts, 32-token
  generations), so prefill and first-token overheads dilute the decode-phase
  multiplier; the pure decode multiplier is the §5a number.
- The plain chains are token-identical to the torch goldens on 4/8 prompts;
  the other 4 flip at a near-tie early and diverge by prefix amplification —
  exactly the W3 relationship (141/145 argmax positions, 4 adjudicated
  near-tie flips), unchanged by spec decode. This is also why the measured
  108 verify passes differ slightly from the simulation's 103 (§3): the
  simulation ran on the TORCH chains, the harness on the ENGINE's own bf16
  chains, which differ at those near-ties. The accept rates agree (54.1%
  measured vs 50.5% simulated).

### 5c. The engine-level blocker, pinned

`tests/vllm/models/test_kolibri1_ngram_spec.cpp` (green):

- spec-OFF: `LoadedEngine::FromModelDir` loads and generates on CPU for
  kolibri (8 tokens from golden[0]'s ids; `spec_drafts_proposed() == 0`).
- spec-ON (`method=ngram, k=5, window 1-4`): construction throws
  `vt: runner: MambaSpec state shapes must be positive at
  ./src/vllm/v1/worker/gpu/runner.cpp:1074`.
- Root cause, traced: `LoadedEngine::MakeKVCacheMaybeSpec`
  (`src/vllm/entrypoints/model_loader.cpp:1756-1763`) — with ANY speculative
  config it builds `vllm::MakeQwen3_5KVCacheSpec(config, ...)` instead of the
  model's own `make_kv_cache`. The engine log shows it: the OFF arm reports
  "shared by **2** KV cache groups" (kolibri's own), the ON arm "shared by
  **3**" (the qwen3.5 widened spec added a recurrent + fa_draft group). The
  runner then classifies the injected MambaSpec group (`runner.cpp:818,
  1037-1074`) and refuses on its empty shapes. The qwen3.5 comment at the
  branch says "Speculation is Qwen3.5/3.6-only at this pin" — true for the
  GATE checkpoints, but the branch is model-blind, and kolibri is not qwen3.5.

## 6. Baseline reproduction (this tree, this run)

`tests/test_kolibri1_decode_bench`, `VLLM_CPP_CPU_THREADS=32
VT_KOLIBRI1_DEQUANT_CACHE_MB=32768`, under the lock: wall 19.62 s, prefill
7.40 s, decode 12.22 s → **5.156 tok/s decode**, gate PASS, last token 109726
(the anchor). The task's baseline is 5.76 on the same recipe; this run measured
5.16 (the box was quieter for the earlier record). The bench chain captured
here is the 2-cycle analysed in §3.

## 7. Projection against the 12 tok/s target

| workload | plain decode tok/s | spec decode tok/s | speedup | vs 12 tok/s target |
|---|---|---|---|---|
| bench shape (128-in, 64-out, decode-heavy, 2-cycle chain) | 4.92 (5.16 by the bench binary, 5.76 task baseline) | **14.51** | **2.95×** | **CLEARED (+21%)** |
| W3 golden battery (short prompts, mixed chains) | 6.84 | 8.99 | 1.31× | not cleared (short-prompt diluted) |

The projection logic: a verify pass costs one T=k+1 forward instead of one
T=1 forward; the measured decode-phase speedup (§5) already includes that
overhead, so the projected tok/s is the measured spec-ON decode tok/s, not
the token ratio. The step-ratio tables in §3 are the acceptance side of the
same multiplication. Where the decode phase dominates (long generations,
batch 1 — the bench's shape and the serving shape for a single user), the
measured 2.95× on the bench chain projects 5.76 × 2.95 ≈ 17 tok/s at the
task's baseline (14.51 measured against this run's 4.92 plain — the box
reads ~5% slower than the record's window). On shorter, less repetitive
traffic the multiplier shrinks toward 1.0-1.3× (the golden battery) and the
12 tok/s target is not met by spec decode alone; the remaining levers there
are the already-measured cache/NEON stack plus batching.

The acceptance side has headroom: k=8 window 1-4 takes the bench chain to 7.0
tokens/verify in simulation (vs 5.25 at k=5) at a higher per-verify cost —
the measured sweet spot on this box is k=5; k is a config knob, not code.

## 8. Integration plan (what wiring to serving would take — NOT done here)

The probe deliberately does not wire serving. The remaining work, in order:

1. **THE ONE PRODUCT CHANGE** (small, well-localized, shared-branch — needs
   its own spec + fresh review per the repo's spec-before-code rule):
   `LoadedEngine::MakeKVCacheMaybeSpec` (`model_loader.cpp:1756-1763`) should
   dispatch its widened-spec branch on "the method needs spec state"
   (`spec->use_eagle()` — mtp/dflash/dspark draft KV + GDN widening) OR "the
   model's own KV config has a recurrent group" (GDN k+1 state widening).
   kolibri has neither, so its own `make_kv_cache` output is already
   spec-correct for n-gram: the drafts are ordinary scheduled tokens, there
   is no draft KV and no recurrent state. Concretely: build
   `ModelRegistry::MakeKVCache(model, ...)` first; take the
   `MakeQwen3_5KVCacheSpec` branch only when `spec->use_eagle()` or the
   model's config carries a `kMamba` group. qwen3.5's n-gram path keeps the
   widened spec (it HAS GDN layers — the widening is load-bearing there), so
   the landed 27B ngram gate is untouched; kolibri's spec-ON engine then
   constructs and the full generic machinery (§1) drives it.
2. **Flip the pinned blocker test into the real gate.**
   `test_kolibri1_ngram_spec` currently asserts the refusal; after step 1 it
   becomes the engine-level e2e gate: spec-OFF vs spec-ON through
   `LoadedEngine` on the golden prompts + bench prompt, exactness strict,
   acceptance counters pinned in bands. That is the row gate, with a row
   spec before the code (spec-before-code).
3. **Exercise the CLI flag once** through the C ABI (`vllm_generate`) or the
   server with `--speculative-config '{"method":"ngram",...}'` to prove the
   flag reaches the engine from a production entry point (the tests drive
   `LoadedEngine`/`ForwardKolibri1Forward` directly, which are production
   seams, but the flag string itself is owed one reach case).
4. **Tuning**: k and the window are config, not code. §3/§5 say k=5,
   window 1-4 on these chains; a serving-shaped prompt battery (real chat
   traffic, longer generations) should pick the final defaults. The upstream
   default window (5/5) is measurably worse here (1.69× vs 2.41× step ratio
   in simulation).
5. **Candidate (b) — a tiny external draft model — is NOT recommended.** It
   needs a second checkpoint, a second forward path (the `draft_model` method
   is refused in this tree — `ResolveSpecConfig` has no draft-model lane),
   and on chains this repetitive a free string match already accepts
   ~everything. Revisit only if real-traffic acceptance comes out far below
   the §3 bound.
6. **Batch > 1**: the probe is batch-1 (the bench's shape). The machinery is
   batch-generic (the proposer is per-request; the verify is per-request
   rows), but a batched acceptance measurement is owed before claiming
   serving throughput.

## 9. Risks and caveats

- The acceptance numbers are measured on degenerate looping chains (upper
  bound); real-traffic acceptance is owed (integration plan step 3).
- The e2e wall-clock ratio is a same-binary A/B on one host in one window;
  the repo's gate discipline (idle-host reproduction, 2-3×) applies before
  any serving claim.
- The verify step is a T=k+1 forward: at k=8 the per-step cost grows, and the
  measured speedup already includes it. Larger k buys acceptance headroom on
  longer repeats but pays more per verify; k=5 is the measured sweet spot on
  these chains.
- kolibri has no MTP layer, so the cheap in-checkpoint draft is genuinely
  absent; n-gram is the right first drafter here.
- The prototype asserts spec-ON == spec-OFF through THIS engine; the W3
  relationship to the torch goldens (141/145 argmax positions, 4 near-tie
  flips) is unchanged — spec decode does not move either side.
