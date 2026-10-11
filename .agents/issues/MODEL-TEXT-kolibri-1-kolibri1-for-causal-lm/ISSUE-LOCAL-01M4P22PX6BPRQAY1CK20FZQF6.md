ID: ISSUE-LOCAL-01M4P22PX6BPRQAY1CK20FZQF6
Title: Probe: speculative decoding feasibility for kolibri1 CPU decode (n-gram draft acceptance measurement)
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-12
Closed: 2026-10-11

## Problem

The CPU kolibri decode bench measures 5.76 tok/s (t32+32GiB cache+NEON) against a 12 tok/s developer target and the measured levers are nearly exhausted. Speculative decoding is the one remaining multiplier that does not change numerics (acceptance is exact: a draft token is accepted iff the target's greedy argmax agrees). kolibri has no MTP/nextn layer, so the cheap draft is absent. Candidates: (a) n-gram/prompt-lookup drafting (llama.cpp -draft style, no trained model), (b) a tiny external draft model. This probe must: (1) inventory the tree's spec-decode machinery and classify model-specific vs generic/portable parts; (2) trace the kolibri CPU decode path and locate the verify hook point; (3) measure n-gram (n=1..4) acceptance potential on the bench greedy chains + W3 goldens; (4) if contained, build a minimal bench-level prototype harness reporting measured accept rate and projected tok/s. Deliverable: feasibility report + measured acceptance estimate committed on row/kolibri-spec-probe; no push, no PR.

## Resolution

PROBE COMPLETE 2026-10-11 (branch row/kolibri-spec-probe, commit pending; report docs/bench-evidence/kolibri1-spec-probe-20261011.md). Findings: (1) PORTABILITY: the tree's n-gram spec-decode machinery is generic end to end — scheduler spec slots (scheduler.cpp:247-251,700-741,948-950,1226-1265), engine draft install/rollback (core.cpp:101-127), runner verify (runner.cpp:3546+ sample_tokens_with_rejection, greedy accept-iff-argmax per rejection_sampler.h:13-51), proposer (runner.cpp:4517-4570 -> ngram_proposer.cpp), and GPUModelRunner is the production runner on CPU (model_loader.h:897). Only the MTP/DFlash/DSpark DRAFTERS and the GDN/mamba spec-state widening are model-specific; n-gram needs neither. ONE blocker: LoadedEngine::MakeKVCacheMaybeSpec (model_loader.cpp:1756-1763) builds MakeQwen3_5KVCacheSpec for ANY spec config, ignoring the model's make_kv_cache; for kolibri it injects a MambaSpec+fa_draft group and the runner refuses at runner.cpp:1074 (pinned by test_kolibri1_ngram_spec, green). (2) MEASURED ACCEPTANCE (exact simulation with the production proposer over the W3 golden chains, pinned by test_kolibri1_ngram_accept, 90/90 green): k=5 window 1-4 -> 103 verify passes vs 248 plain steps (2.408x), 150/297 drafts accepted (50.5%); k=8 -> 2.583x; upstream default window 5/5 is worse (1.687x). Bench chain (128 distinct random prompt, 64-token 2-cycle chain captured from the bench run): k=5 -> 12 passes vs 63 (5.25x), 52/53 accepted (98.1%). (3) PROTOTYPE (contained, bench-level per the brief's fallback since the engine path is blocked by the one seam): test_kolibri1_spec_probe drives the production proposer + ONE production ForwardKolibri1Forward over the k+1 verify sequence on the real checkpoint — bench prompt: spec chain byte-identical to plain, 53/53 accepted (100%), decode 14.51 tok/s vs 4.92 plain (2.95x, CLEARS the 12 tok/s target); W3 goldens: 8/8 byte-identical, 146/270 accepted (54.1%), decode 8.99 vs 6.84 tok/s (1.31x, short-prompt diluted). Green 17/17. (4) Engine reach pinned: spec-OFF engine generates on CPU; spec-ON throws the precise refusal (test_kolibri1_ngram_spec, 6/6 green). (5) INTEGRATION PLAN (not done — serving wiring deliberately out of scope): dispatch MakeKVCacheMaybeSpec's widened branch on use_eagle() or a recurrent group in the model's own KV config (kolibri has none, so its make_kv_cache is already spec-correct for n-gram); then flip the pinned blocker test into the engine-level e2e gate with a row spec; exercise the --speculative-config flag once via the C ABI; tune k/window on serving-shaped traffic; batch>1 acceptance owed. Candidate (b) tiny external draft model NOT recommended. Baseline reproduced: bench 5.156 tok/s this run (task baseline 5.76), anchor 109726 PASS.

INTEGRATION LANDED 2026-10-12 (branch row/kolibri-spec-engine). The probe's
one blocker is fixed and the n-gram proposer now runs in the kolibri ENGINE
path behind the existing opt-in (EngineParams::speculative_config — the same
object the C ABI v6 field and the server's --speculative-config populate).

THE SEAM FIX (the only product change): LoadedEngine::MakeKVCacheMaybeSpec
(src/vllm/entrypoints/model_loader.cpp) now builds the MODEL's own KV spec
(ModelRegistry::MakeKVCache) and takes the widened MakeQwen3_5KVCacheSpec
branch only when spec->use_eagle() OR the model's own groups carry a
recurrent/mamba group. A draft-free n-gram config on a model with no recurrent
group keeps the model's own spec: the drafts are ordinary scheduled tokens, so
the target's groups are the whole cache need and NO draft-proposal group is
added (the fa_draft group is consumed only by the MTP draft head's own paged
KV, runner.cpp:1865-1898, a forward n-gram does not have — and that allocation
is skipped outright on kolibri's multi-cache topology). The default (no spec
config) path is byte-identical: exactly ModelRegistry::MakeKVCache. qwen3.5's
ngram path keeps the widened spec (it HAS GDN layers), so the landed 27B ngram
gate is untouched. RED-FIRST: with the fix reverted, the new seam unit fails
(REQUIRE( 3 == 2 ) — the injected fa+gdn+fa_draft set) and the e2e fails at
construction with the pinned refusal "vt: runner: MambaSpec state shapes must
be positive at runner.cpp:1074" (the spec-OFF arm runs all 9 prompts first,
0 drafts). GREEN with the fix: unit 21/21; e2e 83/83.

NO NEW ENGINE WIRING WAS NEEDED: the scheduler spec slots, the engine's draft
install/rollback, the runner's verify (sample_tokens_with_rejection) and the
draft-free proposer (propose_drafts_ngram) were already generic; the e2e gate
proves the whole loop executed (53/53 drafts accepted on the bench prompt).
tests/vllm/models/test_kolibri1_ngram_spec is flipped from the pinned blocker
into (1) a checkpoint-free seam unit (kolibri under an ngram spec config keeps
its own 2 groups — no MambaSpec, no fa_draft, byte-identical to spec-off —
while qwen3.5 still widens to fa+gdn+fa_draft with the conv row at
(K-1)+k taps and num_speculative_blocks=k) and (2) the engine e2e gate.
tests/capi/test_capi.cpp gains the ABI reach pin: a valid ngram
speculative_config parses and reaches the engine load (MODEL_LOAD on a missing
checkpoint, not INVALID_ARGUMENT).

MEASURED THROUGH THE ENGINE (LoadedEngine, t32 + 32 GiB dequant cache, under
flock, VT_ASYNC_SCHED=0 so BOTH arms run the synchronous scheduler — ngram
forces sync via async_scheduling_compatible()==use_eagle(); with the default
async spec-OFF arm the goldens A/B is confounded by pipelined input prep and
is reported separately):
- EXACTNESS: spec-ON == spec-OFF BYTE-IDENTICAL on 9/9 prompts (the
  deterministic bench prompt 128-in/64-out + the 8 W3 golden prompts). The
  spec-OFF bench chain ends at the pinned anchor 109726.
- ACCEPTANCE (engine counters spec_drafts_accepted/proposed): bench prompt
  53/53 — the pinned number (exact sim over the torch chain: 52/53; the probe
  harness over this tree's bf16 chain: 53/53). Goldens aggregate 146/270 —
  EXACTLY the probe harness's 146/270 (sim over torch chains: 150/297). The
  engine reproduces the harness's acceptance counts exactly, per prompt too.
- ENGINE tok/s (decode phase, first token -> finish): bench prompt 4.34 ->
  13.39 tok/s (3.08x; wall 2.35 -> 5.64, 2.40x) — CLEARS the 12 tok/s target
  through the engine (harness reference: 4.92 -> 14.51, 2.95x). Goldens
  aggregate 6.80 -> 8.17 tok/s (1.20x; harness reference: 6.84 -> 8.99, 1.31x;
  short-prompt diluted, same shape). Default-scheduler run for the record:
  bench 4.32 -> 13.50, goldens 7.50 -> 5.61 (the async spec-OFF arm's pipelined
  input prep is the confound; not the A/B of record).
- W3 VERDICT: unchanged with spec off — test_kolibri1_w3 900/900 PASS (the
  default path is byte-identical; the KV-spec unit pins it too) and the decode
  bench anchor 109726 PASS. With spec ON: byte-identical to spec-OFF (9/9), so
  the torch-golden relationship is identical by transitivity — 4/8 prompts
  token-identical end to end with the same per-prompt shared prefixes the
  probe measured (141/145 argmax positions, 4 adjudicated near-tie flips
  unchanged). No hard flip.
- REGRESSION SURFACE: test_kolibri1_ngram_accept 90/90 (pinned sim stays
  green); test_kolibri1_spec_probe (the harness) green; seam-adjacent suites
  test_loaded_engine_dense 157, test_model_registry 1229,
  test_deepseek_v4_scaffold 684, test_kv_pool_min_fit 16, test_runner 1918,
  test_kv_cache_fp8_wiring 487 — all green (4491 assertions).

REMAINING DELTA (owed, none blocking): (a) the engine's reject-heavy path — on
the most reject-heavy golden (10/84 acceptance) the verify T=1+kr forward
makes spec-ON ~1.8x SLOWER wall than plain on that prompt (10.20 s vs 6.81 s);
the aggregate still wins 1.20x. A per-step cost trace of the reject path
(scheduler rollback + draft re-splice + rejected-slot re-schedule) is the
named follow-up. (b) Batch>1 acceptance (probe plan step 6). (c) Real-traffic
(non-degenerate-chain) acceptance (probe §9 caveat: the measured rates are an
upper bound). (d) Serving-shaped k/window tuning — config, not code
(k=5, window 1-4 measured on these chains; upstream default 5/5 is worse).
