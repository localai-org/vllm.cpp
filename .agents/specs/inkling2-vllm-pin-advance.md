# SPEC — `NEW-CANDIDATE-PORTS` inkling2: vLLM parity-pin advance past PR #69028 (target `ab18334a4`)

- Owning row: `NEW-CANDIDATE-PORTS`.
- Worktree `/tmp/inkling2-pin`, branch `row/NEW-CANDIDATE-PORTS-inkling2-pin`, base `8f6add34b`.
- Kind: record + verification scoping. NOT a full repository pin advance and NOT a
  model port. This unit answers one question and records the answer: is upstream
  vLLM at #69028 a viable PRIMARY oracle for `inkling2_loop`, and what would a
  full pin advance to that revision obligate?

## Trigger / problem

The active parity pin is `e126687a9a` (2026-09-03, vllm#53896), recorded in
[.agents/upstream-sync.md](../upstream-sync.md). Upstream vLLM merged
Inkling-2-Mini support as PR
[#69028](https://github.com/vllm-project/vllm/pull/69028)
(`Inkling2LoopForConditionalGeneration`, `model_type: inkling2_loop`) on
2026-11-21T16:19:34Z, merge commit
`ab18334a4f60da74db7de89fe95594e5ed0cda36` — months AFTER the pin. The landed
W0 spec
[inkling2-loop-w0-gguf-types-and-parser.md](inkling2-loop-w0-gguf-types-and-parser.md)
names the advance as owed. This unit records the reconciliation verdict and the
obligations, and measures gateability as far as this host allows.

## Reconciliation verdict (measured, 2026-11-22)

**Upstream at #69028 IS a viable PRIMARY oracle for `inkling2_loop` in
principle — it is the only reference that implements the architecture
end-to-end — but it is NOT gateable on this CPU-only host.** Evidence, all
fetched at the merge sha:

1. **Architecture coverage.** `vllm/models/inkling2/configs.py` registers
   `Inkling2ModelConfig` (text), `Inkling2MMConfig` / `Inkling2LoopMMConfig`
   (`model_type: "inkling2_loop"`), and `Inkling2DSparkConfig`; the
   `nvidia/` package carries the model, the loop (`loop.py`), MoE, attention,
   sconv, vision and DSpark. The registry maps
   `Inkling2LoopForConditionalGeneration` -> `vllm.models.inkling2`.
2. **Config agreement with our W0 walk.** The HF checkpoint
   `thinkingmachines/Inkling-2-Mini` (ungated, not disabled) declares
   `architectures: ["Inkling2LoopForConditionalGeneration"]`,
   `model_type: "inkling2_loop"`, 24 decoder layers, and
   `loop_config {core_start: 6, core_end: 18, num_passes: 3, gate_rank: 64,
   is_varloop: false}` — the same values our GGUF walk of
   `mudler/Inkling-2-Mini-APEX-GGUF @ d85a5800` mapped (`loop.start`=6,
   `loop.end`=18, `loop.count`=3, `loop.gate_rank`=64, `loop.varying`=false).
   The W0 validation mirror (`0 < core_start < core_end <= num_hidden_layers`,
   `gate_rank >= 1`, >= 2 fixed-loop passes, `attn_res_normalize_values`) is
   byte-for-byte the upstream check at `configs.py:100-138`.
3. **Quantization handling.** `nvidia/quant_config.py` +
   `hf_quant_config.json`: only routed experts are quantized (NVFP4 or MXFP8,
   `parameter_groups` with per-tensor compute/output dtypes); every other
   layer runs unquantized; the LM head runs BF16 with an FP32 output
   (`head_dtype: "float32"` set by default in `Inkling2MMConfig` — the
   annotated f32 exception AGENTS.md asks us to name).
4. **Weights format.** Upstream consumes **HF safetensors only**. The local
   APEX GGUF fixture (`Inkling-2-Mini-APEX-Balanced.gguf`, Q4_1/Q5_1) is NOT
   consumable by the oracle; an oracle run requires the HF checkpoint
   (~26.4B parameters, BF16/F8_E4M3/U8 mix, ~26 GB, single
   `model-00001-of-00001.safetensors` + `mtp.safetensors`).
5. **Platform constraint (the blocker).** Every forward kernel lives under
   `vllm/models/inkling2/nvidia/` with compiled CUDA in `csrc/inkling2/`
   (`conv_mix.cu`, `allreduce_conv_mix.cu`, `seq_parallel.cu`). Kernel
   selection (`nvidia/ops/__init__.py use_sm100_kernels`) branches only on
   device capability: Blackwell sm100 (tml_fa4, tcgen05 attention, TRT-LLM-gen
   expert GEMMs), Hopper (Humming experts), SM12x (Triton experts). There is
   **no CPU path**. A gateability probe needs a fleet GPU
   (`dgx:gpu0`/`thor:gpu0`/`orin:gpu0` per AGENTS.md §Work on a GPU happens
   inside a lease); this host has none.

**Gateability outcome: `gateable = no` at this unit's close.** The exact
blocker: no CPU forward path exists at the target revision, and no GPU lease
was taken in this CPU-only record unit. Owed by local issue
ISSUE-LOCAL (see `## Owed`); the issue files the build+run probe on a leased
GPU as the discharge step. Downloading the 26 GB HF weights was NOT performed:
it would produce an artifact this host cannot execute.

## Obligations ENUMERATED (not reconciled) for a full pin advance to `ab18334a4`

Any future advance of `.agents/upstream-sync.md` past `#69028` must, per
[specs/upstream-pin-advance-e126687.md](upstream-pin-advance-e126687.md),
reconcile ALL of the rows anchored at `e126687a9a`. This unit enumerates them
and reconciles NONE:

1. **The four strict goldens** re-captured at the NEW pin: 27B W4A4,
   32B-NVFP4A16, 35B, Coder. Already owed at `e126687a9a`
   (`specs/upstream-sync-headpin-tokengate.md` `## Owed`); the advance raises
   the bar again.
2. **The OPT-125m declared token-exact gate** re-run at the new pin. It was
   discharged at `e126687a9a` (issue #2794, closed 2026-09-06); a new pin
   voids that discharge.
3. **The BENCHMARK step-6 re-baseline** (#2818, still open per
   upstream-sync.md) against the new oracle, across the five affected rows of
   `vllm-online-serving` and `speculative-decoding`, including the withdrawn
   `nvidia-cutlass-dsl` discharge (C1c).
4. **The `Ported from:` header debt** — at least 177 files naming `55596792`
   and at least 330 naming `e24d1b24` (upstream-sync.md §5.3 floor count) —
   must be re-triaged against the new revision, plus every file ported since
   `e126687a9a`.
5. **Dependency pins re-measured, not transcribed**: the `parity-pin` block in
   upstream-sync.md (vllm runtime/distribution version strings, flashinfer,
   CUTLASS DSL, `transformers` floor — note #69028 uses newer
   `PreTrainedConfig` features such as `has_no_defaults_at_init`, so the
   transformers floor likely moves). Values must come from a measured source
   build, per the `.precompiled` correction (#2896).
6. **Upstream test ports**: `tests/models/inkling2/` (test_model,
   test_attention, test_moe, test_core_ops, test_dspark, test_vision,
   test_allreduce, test_initialization) and the registry/initialization test
   changes ride the sync cycle under the port-tests-in-the-same-change rule.
7. **New upstream surfaces the PR adds that our rows must at least census**:
   the DSpark speculator (`inkling2_dspark`, speculative config changes), the
   vision tower, the Rust chat backend/renderer files upstream now carries,
   and `routed_experts_capturer.py`.
8. **Oracle file**: `.agents/oracles/vllm.md` pointer, and if inkling2 gets a
   dedicated secondary path, its llama.cpp hazard pins
   (`690b87d` floor, #32710, #32755) stay as recorded in the W0 spec — a vLLM
   pin advance does not touch them.

## Scope

IN:
1. This spec (reconciliation verdict + obligations + verification plan).
2. A local issue owning the GPU gateability probe, filed with
   `scripts/agent-issue.py create --row NEW-CANDIDATE-PORTS`.
3. The W0 spec's `## Now` updated to name this unit's outcome.
4. Bench evidence under `docs/bench-evidence/new-candidate-inkling2-pin-20261122/`
   (flat) recording the fetch commands and the raw config/quant facts quoted above.

OUT: any edit to `.agents/upstream-sync.md` or the `parity-pin` block (that is
a full sync cycle, gated by the obligations above); any model, kernel, or
parser code; any GPU work; the HF weights download; the llama.cpp pin floor
(W0 owns it); reconciling any of the enumerated obligations.

## Verification plan (how the probe discharges gateability later)

1. Take a GPU lease (`rc run`/`rc hold`; never ssh+flock). Build vLLM from
   source at `ab18334a4f60da74db7de89fe95594e5ed0cda36` in `/tmp` per
   environment.md (CUDA toolkit install, venv under `/workspace`, `-j 4`,
   `cp -rL` out of CIFS).
2. Download `thinkingmachines/Inkling-2-Mini` (HF safetensors, ~26 GB) and
   record its sha256s beside the run recipe.
3. **Run, not construct**: `vllm serve` or `LLM(...)` generate on a small
   prompt at greedy. A successful config construction proves nothing.
4. Record the measured `parity-pin` identity strings (runtime + distribution
   version, flashinfer) — those feed the real sync cycle, which remains a
   separate gated unit.

## Tests

This is a records unit: no product code, so no new unit tests. The applicable
checks are the record gates: `scripts/check-agent-record.py --report` rot 0,
`scripts/agent-issue-index.py --refresh` clean, and
`scripts/agent-preflight.sh --staged` exit 0.

## Risks / decisions

- The verdict "viable but not gateable here" is deliberately two-valued: a
  future reader must not read `gateable = no` as "the oracle is wrong" nor as
  "viable" alone as "measured".
- The GGUF fixture and the HF oracle weights are different artifacts of the
  same architecture; parity between our GGUF arm and the oracle requires
  either an HF-consuming arm of ours or a quant-matched GGUF comparison via
  llama.cpp (`llama-cpp` oracle, floor `690b87d`). That choice belongs to the
  port row, not this unit.
- Upstream's ops are per-SM-family; a single-GPU probe validates one family.
  The obligations list does not assume cross-family parity from one probe.

## Stop conditions

`NEEDS_DECISION` if the developer wants this unit upgraded to the full sync
cycle (then upstream-sync.md edits and the golden re-captures move in scope).
Stop and report if `thinkingmachines/Inkling-2-Mini` becomes gated or
disabled before the GPU probe runs; the probe recipe then needs a new weights
source recorded first.

## Owed

- The GPU gateability probe at `ab18334a4` (build + run Inkling-2-Mini), owned
  by the local issue filed in this change; discharges `gateable = no`.
- The full pin advance to `ab18334a4` with every obligation in the ENUMERATED
  list reconciled — a separate gated sync-cycle unit.

## Git integration

One PR carrying spec + issue + W0 `## Now` update + evidence; push
`row/NEW-CANDIDATE-PORTS-inkling2-pin` to `mine`; human developer merges.
