# llama.cpp — the CPU and GGUF k-quant floor

Never the mirror source: llama.cpp's structure is not vLLM's, and a behavior
difference between them is settled by vLLM. What it supplies is a **floor** —
the CPU and GGUF k-quant speed and memory numbers a user can actually get today,
which is the honest denominator on every path where vLLM's own CPU support is
not the thing being compared.

The pin is **stock upstream**, tag `b10451`. That is what the sentence above
requires. A user installs a release, so the floor is a release.

**This oracle is gateable as of 2026-08-22, and the measurement is recorded
rather than assumed.** Stock upstream at this pin was built and run inside an
`rc` lease on `thor:gpu0`: it fetched from `ggml-org/llama.cpp`, built CPU-only
from a clean tree at the exact object, loaded the recorded Qwen3.8-27B Q4_K_M
artifact and generated coherent text. The identity chain, the build and run
recipes and the output are in
[`../../docs/bench-evidence/oracle-llamacpp-b10451-gateable-20260822.md`](../../docs/bench-evidence/oracle-llamacpp-b10451-gateable-20260822.md),
which is what `evidence` now names. `AGENTS.md` admits `gateable = yes` only
after an oracle demonstrably builds and runs the model, and
`scripts/check-oracle-pins.py` refuses `gateable = yes` beside an `#N`, so the
flag and the evidence had to move together.
[#857](https://github.com/mudler/vllm.cpp/issues/857) is discharged.

**Gateable is not measured.** The flag says the oracle runs. It says nothing
about any number. Every floor in this tree is still owed by
[#1003](https://github.com/mudler/vllm.cpp/issues/1003), and the gateability run
deliberately produced none: one repetition, a six-token prompt, no clock pinning
(no job in a lease can pin a clock), no contention control. Do not promote a
timing out of that evidence file.

**One fidelity fact from the run that a comparison must carry.** At `b10451`
llama.cpp loads 851 of the artifact's 866 tensors and ignores all 15 tensors of
`blk.64`, 289,527,808 bytes, four of which are the `nextn.*` multi-token
prediction head. So this oracle runs 64 layers and no MTP head on this
checkpoint. "Quant-matched against the same weights" is therefore not
automatically matched work: an arm that runs block 64 does strictly more per
token than this denominator. State which side ran what, or the ratio measures a
configuration difference and reads as a performance one.

**Every recorded llama.cpp number in this tree predates this pin and was taken
against something else.** From 2026-07-22 to 2026-08-16 this record pinned
`237ad9b96`, a local-only commit on branch `localai-paged` in the developer's
checkout, 65 of our own performance commits past upstream tag `b9827`, built from
a working tree carrying 27 uncommitted entries. Six of those 65 commits change
`ggml/src/ggml-cpu/`, and `570aadd7a` emits a fused Gated Delta Net op and a
discriminated SSM_CONV decode op default-on with CPU reference implementations
that stock does not have, so the CPU arm was affected and not only the CUDA arm.
The label `b9892` was derived from `git rev-list --count`, and upstream tag
`b9892` is a real, different object at `ee445f93d`.

**The affected measurements are enumerated, with a re-take verdict each, in
[`../specs/oracle-llamacpp-repin-stock.md`](../specs/oracle-llamacpp-repin-stock.md).
That spec is the one source of truth for the set, and this file deliberately does
not restate its size.** It used to say "all nine" while that spec listed twelve,
and then thirteen. A count of one file kept inside another goes stale the moment
the first file grows, which `AGENTS.md` §Records names as the coupling to avoid,
and this is the first surface a reader of the pin reaches. The set has grown
three times, each time because a sweep instrument was widened rather than because
anything new was measured, so any number you remember is a lower bound.

That spec also enumerates the distinct llama.cpp revisions those measurements ran
against, and one of them is a fork **branch** with no commit recorded anywhere in
this tree. Read the count there rather than here, for the reason above and for a
sharper one: that spec's own `## Owed` already schedules another entry, so a
number written here would be stale against a change it can already see coming.

[#1003](https://github.com/mudler/vllm.cpp/issues/1003) owes the re-take. The
superseded measurements stay where they are, with their provenance, including
[`../specs/cpu-llamacpp-floor-remeasure-2026-07-22.md`](../specs/cpu-llamacpp-floor-remeasure-2026-07-22.md)
and the A76 dot-product, elementwise-GEMM, GDN-orientation and threadpool specs.

**This oracle's greedy decode is NOT deterministic across its own supported
kernel paths, so a gate must pin the EXECUTED PATH and not only the revision.**
Measured 2026-09-02 on `thor:gpu0` (aarch64, `NEON=1 MATMUL_INT8=1 SVE=1
DOTPROD=1 REPACK=1`), rc job `deb6322d-bd06-4dd1-a5ac-2dec9987fbe1`. Stock
`b10451`, one artifact, one recipe, greedy, run twice with the only difference
being `use_extra_bufts` — the field the STOCK `-nr/--no-repack` flag sets
(`common/arg.cpp:2411-2418`, `LLAMA_ARG_REPACK`; `common/common.cpp:1669` feeds
`make_cpu_buft_list`, `src/llama-model.cpp:910,942,1304`, default true). Both
arms are supported stock configurations, and they emit different greedy tokens:

```text
ORACLE_SELF_DIVERGENCES=1/6
prompt 1, index 34: repack ON -> 3095,  repack OFF -> 198
```

Teacher-forcing one arm along the other's ids and diffing all 71,516,160 final
logits, the per-step max abs logit delta is **min 0.2020, median 0.3790, max
1.3657** over 288 steps (rms min 0.0412, median 0.0729, max 0.1856). The repack
is a byte permutation of the same quantized values (`make_block_q4_Kx8`,
`ggml/src/ggml-cpu/repack.cpp:2836-2870`), so the dequantized weights are
identical and only the order and granularity of the fp32 arithmetic differs.
`b10451` has four distinct fp32 rounding schedules for a Q4_K matvec — the
aarch64 repacked gemv, the aarch64 NEON/SVE vec_dot, the portable `_generic`
vec_dot, and the generic repacked gemv — and which one runs is decided by host
architecture, `-mcpu` feature set, repack state and batch size.

**Three stock levers move this oracle's greedy tokens, and one does not.**
Measured 2026-09-02 on `thor:gpu0`, rc job
`8480a30e-0d6d-44a7-b1b4-00e9d36c888d`, twelve runs of the stock oracle over one
artifact and one recipe, ten of them teacher-forced along the stock default's own
ids so every arm's argmax describes the identical preceding context. Evidence:
[`../../docs/bench-evidence/qwen38-27b-q4km-oracle-path-pin-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-oracle-path-pin-20260902.md).

| lever | stock flag | moves the tokens? |
|---|---|---|
| `use_extra_bufts` | `-nr/--no-repack` | yes, 1 of 6 prompts |
| `n_ubatch = 1` | `-ub/--ubatch-size` | yes, 1 of 6 prompts; `-ub 4` moves nothing at these prompt lengths |
| flash attention off | `-fa/--flash-attn` | yes, 2 of 6 prompts |
| `n_threads` | `-t/--threads` | **no**: 1, 4 and 13 threads are byte-identical to 14 over all 288 steps |

`n_ubatch` matters because `forward_mul_mat_one_chunk` calls `gemm` only when
`nrows > 3` and `gemv` for the tail (`ggml/src/ggml-cpu/repack.cpp:4240`), so an
ubatch of 1 sends every prefill row through the GEMV body, and the prefill writes
the KV cache every later step reads. `n_threads` does not, because
`repack.cpp:4317-4372` lets a chunk boundary select WHICH rows a thread computes
while each row's dot product completes inside one kernel call over the full
`ne00`.

**`flash_attn_type` defaults to `AUTO`, and on this host `AUTO` resolves to
`ENABLED`** -- the explicit `enabled` arm is byte-identical to the default over
all 288 steps and the `disabled` arm is not. A record that names the revision and
the repack state but not the attention kernel is still under-specified.

**What this obliges of any token gate against this oracle on the CPU tier.**
Record the executed path, not just the pin: assert and record
`use_extra_bufts`, `n_ubatch`, and the RESOLVED flash-attention type (not the
`AUTO` request), together with the `system_info` capability line, the host
architecture, the feature set the binary was compiled for, and the thread count
-- the last for completeness, since it is measured not to matter here. A gate that pins only the revision is
under-specified, and a 6-of-6 token-exactness demand at a sub-0.2-logit margin is
asking for bit-reproduction of one specific kernel rather than for arithmetic
quality — `b10451` scores 5 of 6 against itself when asked.

**This is not a licence to excuse a divergence against this oracle.** Being
inside the oracle's noise band per divergence is not the same as being at its
noise floor by rate: that perturbation flips 1 of 6 prompts, and an engine that
flips more than that carries an additional term of its own. Evidence and the
worked case:
[`../../docs/bench-evidence/qwen38-27b-q4km-oracle-self-consistency-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-oracle-self-consistency-20260902.md).

**Assert the tree, not only the commit.** A pin names a commit, and a commit
cannot tell you what was built. The measurements above came from a directory
somebody develops in. Before any number is recorded against this oracle, either
build from a fresh `git archive` or a fresh clone of the pinned SHA, or assert
`git status --porcelain` empty on the source tree and record that assertion
beside the number. Record the built binary's sha256 either way. The 2026-08-22
gateability run satisfies this for its own binaries and for nothing else: it
fetched `--depth 1` into an empty directory, asserted `git status --porcelain`
empty at 0 bytes, and recorded both sha256 values. `GGML_NATIVE=ON`, so a
recorded binary sha256 identifies a build on a named host and never a tree.

**`llama-cli` at `b10451` requires `-st` for a clean exit.** After
generating `-n` tokens, `llama-cli` enters conversation mode and writes `>`
prompts to stdout in a tight loop. `< /dev/null` does not prevent this.
`--no-conversation` and `--simple-io` do not prevent it either. Only
`-st`/`--single-turn` causes a clean exit after generation. Without it, one run
on `strix:gpu0` produced a 6.2 GB log file before the process was killed. Assert
`-st` on every `llama-cli` invocation at this pin.

```oracle-pin
id = llama-cpp
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = CPU and GGUF k-quant speed and memory floors, quant-matched against the same weights
pin = 11fe02151f79c41d0d4af7da708755d73b9c0da6
pin_label = llama.cpp main, 2026-10-04
pinned_on = 2026-10-03
gateable = yes
evidence = docs/bench-evidence/oracle-llamacpp-11fe0215-gateable-20261003.md
```

**Prior pin: `10bf611e533d81f739128304991c5e133c6aebd8` (`b10451`),
pinned 2026-08-16, gateable evidence
[oracle-llamacpp-b10451-gateable-20260822.md](../../docs/bench-evidence/oracle-llamacpp-b10451-gateable-20260822.md).**
Every gate recorded against `b10451` stays attributed to it. The pin advanced
on 2026-10-03 because `b10451` REFUSES the unsloth `Qwen3.8-27B-Q4_K_M.gguf`
artifact (`missing tensor 'blk.64.ssm_conv1d.weight'`; the artifact's `blk.64`
MTP block carries no SSM tensors and the `qwen35.nextn_predict_layers` KV
override is inert at the pin), which left the TT-KEEPQUANT-INT8DOT flip band
gate without a denominator
([tt-int8dot-flip-live-gates-20261003.md](../../docs/bench-evidence/tt-int8dot-flip-live-gates-20261003.md)).
`11fe0215` LOADS and generates on the same artifact (MTP export gap fixed
upstream). 27B Q4_K_M band-gate denominators from 2026-10-03 on are
`11fe0215`, not `b10451`.
