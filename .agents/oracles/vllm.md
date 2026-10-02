# vLLM — the primary oracle

The mirror source and the only oracle that outranks the others. Where vLLM
implements a behavior, it defines it; a secondary oracle disagreeing with vLLM
loses.

**The pin lives in [`../upstream-sync.md`](../upstream-sync.md)**, in its
` ```parity-pin ` block, together with the runtime and distribution version
strings a live oracle reports about itself and the `+g<sha>` constraint
`assert_oracle_commit` enforces. It is restated below only as identity — the
sync cycle advances the block over there, and `tools/bench/` reads that block,
not this file.

```oracle-pin
id = vllm
role = primary
upstream = https://github.com/vllm-project/vllm
scope = every behavior vLLM implements — defaults, modes, errors, edge cases, and both correctness and speed gates
pin = a7c23ac96d7806e7c7e7d862eadbce5a33529b94
pin_label = 0.3.0.dev267
pinned_on = 2026-09-26
gateable = yes
evidence = .agents/sync/2026-09-22-a7c23ac96d.md
```

## What this pin establishes, and what it does NOT

**`gateable = yes` above says the oracle builds and runs. It does NOT say any
gate in this tree has been run against it.** Those are different statements and
this section keeps them apart, because the pin advanced
([#2817](https://github.com/mudler/vllm.cpp/issues/2817)) on a developer ruling
that put step 6 after step 7, so the advance carries obligations it has not
discharged.

### Established, on two boards

**This heading used to read "and this is the whole of it", and it named only the
Thor build.** The 2026-09-04 capture on `dgx:gpu0` establishes strictly more —
item 2 below and the `dgx:gpu0` row in `## Device-scoped gateability` — so the
claim of wholeness was false the moment those landed. It is corrected here rather
than left for a reader to reconcile against the section that falsifies it.

**On `dgx:gpu0` (GB10, `sm_121a`), 2026-09-04**, job `7386f034`: builds from
source at this revision, runs `facebook/opt-125m`, and reproduces the committed
golden `tests/parity/goldens/opt_greedy/greedy_ids.npy` under the declared
`#2794` recipe — `TOKENGATE_VERDICT PASS`. Item 2 states the numbers and the
narrowness. That is the strongest thing established at this pin.

**On `thor:gpu0`**, `e126687a9a828d513c01a07cd69f025f27d63280` (2026-08-31,
vllm#53896, the revision that registers `Qwen4ExpForCausalLM`) **builds from
source and runs a model**. Measured 2026-09-03, aarch64, NVIDIA Thor, compute
capability 11.0, driver 595.78:

```console
SRCBUILD_RC=0   94 min, MAX_JOBS=4, TORCH_CUDA_ARCH_LIST=11.0, CUDA 13.0.88
EXT_PRESENT=True  seven compiled extensions; vllm._custom_ops.rms_norm EXECUTED on device
IMPORT_RC=0     vllm.__version__ = 0.28.1rc1.dev132+ge126687a9, read from cd /
RUN_RC=0        facebook/opt-125m, greedy, FLASH_ATTN/FA2, eager AND compiled
```

Evidence: [`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md),
issue [#2611](https://github.com/mudler/vllm.cpp/issues/2611), and the install
half [`../sync/2026-09-02-e126687.md`](../sync/2026-09-02-e126687.md)
([#2593](https://github.com/mudler/vllm.cpp/issues/2593)).

### NOT established, and the pin's movement makes none of it true

1. **`qwen4_exp` does not run on this fleet at this revision.** The model this
   pin was taken for: its QSA indexer's `cooperative_topk` refuses to launch with
   a cluster misconfiguration
   ([#2626](https://github.com/mudler/vllm.cpp/issues/2626), cause
   unestablished), and its published safetensors arms exceed the largest fleet
   box. So `MODEL-MM-QWEN4-EXP` has a registered, importable, executable vLLM
   that still cannot serve its own model here. That is strictly better than the
   prior pin, where the architecture was not registered at all, and it is not a
   runnable oracle for the row.
2. **One gate model, one gate, four goldens still owed.** The declared
   token-exact gate at this pin is specified by
   [`../specs/upstream-sync-headpin-tokengate.md`](../specs/upstream-sync-headpin-tokengate.md)
   ([#2794](https://github.com/mudler/vllm.cpp/issues/2794)). **It was captured
   on 2026-09-04 and it PASSED**, on `dgx:gpu0` (GB10, `sm_121a`), the device the
   committed golden came from, with `--runs 5`, by job
   `7386f034-246a-4af5-9a04-f98aafffce54` against a source build at this
   revision: `IDS mismatched_positions 0 of 96`, `IDS_BYTE_EQUAL True`,
   `SELECTOR K=5 multi_valued_cells 0`, `TOKENGATE_VERDICT PASS`
   ([`../../docs/bench-evidence/opt125m-token-gate-e126687-dgx-20260904.md`](../../docs/bench-evidence/opt125m-token-gate-e126687-dgx-20260904.md),
   [`../sync/2026-09-05-e126687-pingate.md`](../sync/2026-09-05-e126687-pingate.md)).
   That capture also answers the OPT golden that was never re-validated at the
   PRIOR pin either. This item used to read "No gate model, and no gate", and
   "none has been re-validated here"; the capture falsifies both, and the second
   sentence is now false only of the REMAINING goldens.

   **What is still owed is the other four strict goldens** — 27B W4A4,
   32B-NVFP4A16, 35B, Coder. The pin advance re-captured all four at
   `5559679229`; none has been re-captured at `e126687a9a`. That obligation is
   NOT anchored on #2794, which the capture closes; it lives under `## Owed` in
   the tokengate spec, and whoever takes it files the issue then. Read narrowly,
   this bar is the OPT-125m ORACLE gate at six prompts and sixteen tokens: it
   establishes that the pinned oracle reproduces the committed golden on this
   board, not that the fleet's production checkpoints do.
3. **Step 6 is owed POST-HOC**
   ([#2818](https://github.com/mudler/vllm.cpp/issues/2818)). Narrowed by
   [`../sync/2026-09-03-e126687-step6.md`](../sync/2026-09-03-e126687-step6.md)
   ([#2771](https://github.com/mudler/vllm.cpp/issues/2771)) to **FlashInfer
   `0.6.15.post1` to `0.6.18` on two gates**: `vllm-online-serving` (three
   throughput rows plus the startup row), where FlashInfer is the NVFP4 GEMM
   under the denominator and the CUTLASS source tree our own arm compiles
   against, and `speculative-decoding` (two rows), where it is the oracle's
   attention backend. **`nvidia-cutlass-dsl` `4.6.0` to `4.6.2` is NOT
   discharged**: a fresh review of
   [#2783](https://github.com/mudler/vllm.cpp/pull/2783) found a warmup path
   gated on `has_device_capability(90)`, which is `>=` and admits `sm_121`, whose
   `compile(...)` is unguarded — it cannot move the steady-state math, but it can
   abort engine start and it sits inside the startup ratio
   `docs/benchmarks/vllm-online-serving.md:73` publishes.
   The `transformers` floor (`>= 5.5.3` to `>= 5.10.4`, resolved `5.14.1`, above
   both) and the `VLLM_ALLREDUCE_USE_FLASHINFER` default flip (inert at
   `tensor_parallel_size == 1`, which every committed gate runs) were discharged
   at the PRIOR pin and **still hold at this one**, because neither argument
   depends on which revision is pinned. **A red on the re-measurement requires
   reverting this pin**, not holding it; that is part of the developer ruling and
   is recorded in [`../upstream-sync.md`](../upstream-sync.md).
4. **`dgx:gpu0` has been read at this pin, twice, and neither reading is a
   benchmark.** This item used to say "No reading on `dgx:gpu0`. Only
   `thor:gpu0` was measured, on one day." That is no longer true. Job
   `7386f034` built the oracle from source on `dgx:gpu0` on 2026-09-04 and passed
   the OPT token gate on it (item 2 above); job
   `d7908816-96a3-40ed-8024-c3ad0cc34d77` read the installed package's own
   metadata there on 2026-09-05, `VERSION_READ_RC=0` at its `step35.log:73` with
   the `metadata_vllm=` block at `:75`
   ([`../sync/2026-09-05-e126687-pingate.md`](../sync/2026-09-05-e126687-pingate.md)
   §2 item 3). **Not job `8c4f639c`, which this item credited until now**: that
   job's own read returned `VERSION_READ_RC=1` with
   `ModuleNotFoundError: No module named 'vllm'` — its wheel install had already
   failed, `INSTALL_VLLM_RC=1` at `stepA.log:912` — and its `succeeded`/`exit 0`
   attests the job and not the step inside it.
   [`../sync/2026-09-05-e126687-step6-c1a.md`](../sync/2026-09-05-e126687-step6-c1a.md)
   still names the wrong job; it belongs to another wave and
   [#2997](https://github.com/mudler/vllm.cpp/issues/2997) owes its repair.
   **What is still absent is a benchmark reading.** Every binding number in this
   tree was taken on GB10 at the PRIOR pin, none has been re-taken at this one,
   and a build-and-identity reading does not transfer to a throughput row.
5. **Step 5 did not run.** The 290-entry PORT-NOW queue for
   `5559679229..e126687a9a` is classified and reconciled but unworked, so **at
   least 177** files whose `Ported from:` header names `55596792` now name a
   revision BELOW the pin. That is a FLOOR — the headers wrap, so no single
   probe sees all of them; see
   [`../sync/2026-09-03-e126687-advance.md`](../sync/2026-09-03-e126687-advance.md)
   §5.3. Owed by [#2611](https://github.com/mudler/vllm.cpp/issues/2611).

**Evidence for the limits above.**
[#2626](https://github.com/mudler/vllm.cpp/issues/2626) is owned by
`MODEL-MM-QWEN4-EXP` and listed under `## Owed` in
[`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md); the
measurement and its explicit non-claims are in
[`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md)
§6 and §7.

### The prior pin, for anyone reading a number taken under it

`5559679229bc961848b121ccdeaa8fa5d79bec98`, `0.26.0.dev0`, pinned 2026-07-26,
FlashInfer `0.6.15.post1`, CUTLASS DSL `4.6.0`, transformers `5.14.1`. Every
binding ratio published in `docs/benchmarks/` was measured against it. That
advance re-captured goldens on its own oracle and recorded zero real drift; this
one has re-captured ONE of five, which is the difference §"NOT established"
item 2 is about.

## Device-scoped gateability

`gateable = yes` above is a property of the oracle, not a promise about every
board. Where a device has been MEASURED to build and run a gate model, it is
recorded here with the evidence; absence from this table means unmeasured, never
unsupported. One row per measurement, appended by the change that made it.

**AT THE CURRENT PIN THIS TABLE NOW HAS EXACTLY ONE ROW, and its narrowness is
the honest reading of it.** The `dgx:gpu0` row is the 2026-09-04 token gate: a
source build at `e126687a9a` served `facebook/opt-125m` and reproduced the
committed golden byte-for-byte. That answers "does a gate model run at
`e126687a9a`" on one device with one model, and it answers nothing about the
production checkpoints the binding rows use. **This paragraph used to say the
table was empty and that nothing in this file answered that question; the capture
falsified it.**

The two `strix:gpu0` rows were measured at the PRIOR pin `5559679229`: the first
row's evidence file records `SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0.dev0+g5559679229`
at `:246`, `vllm.__version__ = 0.26.0.dev0+g5559679229` at `:249` and the same
string in the engine banner at `:340`; the second row's records
`VLLM_VERSION = 0.26.0.dev0+g5559679229` in its build-identity block. They are
kept, with their pin column, because they are real measurements of a real board
and deleting them would destroy evidence — but they do not answer the question at
the CURRENT pin.

`thor:gpu0` is still deliberately NOT added, and the reason is NOT that it
skipped the golden. It compared: `2026-09-03-e126687-runhalf.md:214-221` records
`All 96 token ids equal tests/parity/goldens/opt_greedy/greedy_ids.npy exactly,
on both legs`, `EXACT_MATCH_vs_pin_golden True  mismatched_positions 0` under
eager and under compiled, reproduced on a second lease. **An earlier version of
this paragraph said that run had "no golden comparison"; that is false, and the
discriminator it stated does not discriminate.** The real one is in the next
paragraph of the same file, which tells the reader how to take its own result:

> **Read that as informative and not as a gate.** The golden was captured at the
> **pin**, 1465 commits earlier, on `dgx` (sm_121a), against a bf16-materialized
> checkpoint; this is the **candidate** on Thor (sm_110) against the raw fp16
> checkpoint that vLLM rounds at load. Agreement across that many differences is
> worth recording and is not a parity result, because nothing here was designed
> as one and one battery of six prompts at 16 tokens gates nothing.

The `dgx:gpu0` row is admitted because it holds every one of those differences
fixed: the declared `#2794` recipe (`scripts/opt-oracle-capture.py`, `--runs 5`)
on the board the golden came from, against the same bf16-materialized
checkpoint. Thor's agreement is a stronger fact than a skipped comparison and it
is still not this table's subject, which is a gate measured under the conditions
its golden was captured under.

| device | arch | pin measured at | measured | builds | runs a gate model | evidence |
|---|---|---|---|---|---|---|
| `dgx:gpu0` | GB10, `sm_121a` (CUDA 13.0, driver 580.173.02) | **`e126687a9a`, the CURRENT pin** | 2026-09-04 | yes — from source, `PORCELAIN_LINES=0`, wheel `vllm-0.28.1rc1.dev132+ge126687a9-…-linux_aarch64.whl` | yes — `facebook/opt-125m` bf16, 6 prompts x 16 greedy tokens, `--runs 5`: `IDS mismatched_positions 0 of 96`, `IDS_BYTE_EQUAL True`, `TOKENGATE_VERDICT PASS`. The narrowest gate model in the tree; says nothing about the production checkpoints | [`opt125m-token-gate-e126687-dgx-20260904.md`](../../docs/bench-evidence/opt125m-token-gate-e126687-dgx-20260904.md) |
| `strix:gpu0` | `gfx1151` (RDNA 3.5, Radeon 8060S, ROCm 7.2.4) | **`5559679229`, the PRIOR pin** | 2026-09-03 | yes | yes — Qwen3.8-27B Q4_K_M GGUF, 6 prompts x 48 greedy tokens, reproducible | [`oracle-vllm-gfx1151-20260903.md`](../../docs/bench-evidence/oracle-vllm-gfx1151-20260903.md) |
| `strix:gpu0` | `gfx1151` (RDNA 3.5, Radeon 8060S, ROCm 7.2.4) | **`5559679229`, the PRIOR pin** | 2026-09-03 | yes | yes — and SCORED a gate: `prompt_logprobs` teacher-forcing over 6 x 48 steps, both configurations, negative control discriminating at 21.24 nats | [`q4km-neartie-vllm-oracle-20260903.md`](../../docs/bench-evidence/q4km-neartie-vllm-oracle-20260903.md) |

**A residual this table exists to hold, and does not yet close.** Every checker
reads the `gateable` field, not the prose above it, and no field names the device
or the model a `yes` was measured on. So `gateable = yes` is as strong as the
best row here. **This paragraph used to end "and the best row here is at the
prior pin. Whoever adds the first row at `e126687a9a` — the token gate on
`dgx:gpu0` ([#2794](https://github.com/mudler/vllm.cpp/issues/2794)) is the
obvious candidate — closes that gap for one board and no more", and the table
directly above it now falsifies that.** The `dgx:gpu0` row IS at `e126687a9a`,
captured 2026-09-04, and #2794 closed `COMPLETED` on 2026-09-06
(`gh issue view 2794 --repo mudler/vllm.cpp --json state,closedAt,stateReason`,
read 2026-09-12). The residual is therefore narrower and still open: the gap is
closed for ONE board and ONE gate model, `facebook/opt-125m` at six prompts and
sixteen tokens. It is not closed for `thor:gpu0`, for `strix:gpu0`, or for any
production checkpoint at this pin, and no field records that limit. Whoever adds
the next row at `e126687a9a` narrows it further; nobody closes it by reading the
field.

**gfx1151 needs five packages a bare ROCm image does not carry**, and each of
their absences presents as a device failure rather than as a provisioning gap:
`python3-dev`, `rocm-libs`, `libdrm-dev`, the ROCm `torchvision`, and `amdsmi`.
`amdsmi` is the one that matters most: `vllm/platforms/__init__.py:110-128`
decides whether the platform is ROCm by importing it, so without it vLLM falls
back to `UnspecifiedPlatform` on a fully working ROCm box. The evidence file
carries the exact message each one produces.

**The oracle is not self-consistent on this device, and the gate a reader picks
decides the answer.** Its two supported configurations are each byte-identical
to their own repeat and still disagree with each other: 2 of 6 prompts
free-running, and teacher-forced each one diverges from its *own* recorded
decode, 3 of 288 under compiled and 6 of 288 under eager. That floor is a
property of the oracle on this board, not of anything measured against it, so a
score of 0 divergences here is below the instrument's own floor and must be read
with it. `AGENTS.md` §Gates admits an explicitly ratified distributional gate
when an oracle's greedy decode is non-deterministic; that precondition is now
MEASURED for the primary oracle, and the ratification for the Q4_K_M ROCm arm is
still open under [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
Measured by [#2809](https://github.com/mudler/vllm.cpp/issues/2809):
[`q4km-neartie-vllm-oracle-20260903.md`](../../docs/bench-evidence/q4km-neartie-vllm-oracle-20260903.md).

**No `HSA_OVERRIDE_GFX_VERSION` was set for that measurement**, and none may be
set for another. That knob makes the runtime report a different device, and a
pin taken under it is a pin on a fiction.
